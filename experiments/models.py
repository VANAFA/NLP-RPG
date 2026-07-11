"""Model loading + generation for Experiment 1.

Only two underlying weight sets are ever loaded — the base model and the
LoRA-adapted (finetuned) model — since the "steered" vs "unsteered" configs
share the same weights and just toggle a forward hook (steering.py) around
generation, rather than needing four separate model instances.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import config as cfg
import steering


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    layer_idx: int
    steering_vector: Optional[torch.Tensor]


def _load_contrastive_pairs():
    with open(cfg.CONTRASTIVE_PAIRS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _finish_loading(model, tokenizer) -> LoadedModel:
    model.eval()
    layer_idx = steering.pick_layer_index(model, cfg.STEERING_LAYER_FRACTION)
    vector = steering.compute_steering_vector(model, tokenizer, layer_idx, _load_contrastive_pairs())
    print(f"  -> steering layer {layer_idx}, vector norm-before-alpha=1.0 (unit), alpha={cfg.STEERING_ALPHA}")
    return LoadedModel(model=model, tokenizer=tokenizer, layer_idx=layer_idx, steering_vector=vector)


def load_base() -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return _finish_loading(model, tokenizer)


def load_finetuned() -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, cfg.LORA_ADAPTER_ID)
    return _finish_loading(model, tokenizer)


def generate_chat(
    loaded: LoadedModel,
    messages: list,
    use_steering: bool,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    """Core generation call: `messages` is a full chat history (system +
    any number of alternating user/assistant turns), so this also serves
    Experiment 2's growing multi-turn campaign log — `generate()` below is
    just the single-turn (system, user) case Experiment 1 needs."""
    tokenizer = loaded.tokenizer
    device = next(loaded.model.parameters()).device
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    vector = loaded.steering_vector if use_steering else None
    max_new_tokens = cfg.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    temperature = cfg.TEMPERATURE if temperature is None else temperature

    start = time.perf_counter()
    with torch.no_grad(), steering.SteeringController(loaded.model, loaded.layer_idx, vector, cfg.STEERING_ALPHA):
        output_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=cfg.TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    hit_max_tokens = bool(
        new_tokens.shape[0] >= max_new_tokens and new_tokens[-1].item() != tokenizer.eos_token_id
    )
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, latency_ms, hit_max_tokens


def generate(loaded: LoadedModel, system_prompt: str, user_text: str, use_steering: bool):
    return generate_chat(
        loaded,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        use_steering,
    )
