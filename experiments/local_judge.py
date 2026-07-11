"""Local, no-API-key fallback for the LLM-as-a-Judge role (config.JUDGE_BACKEND
= "local"). Used when no ANTHROPIC_API_KEY is available.

Uses a separate, larger local model (config.LOCAL_JUDGE_MODEL_ID, the same
Qwen3-4B-Instruct the actual game backend runs — see backend/server.py) with
schema-constrained decoding (lm-format-enforcer, same technique as
backend/server.py) so output is always valid JSON matching the requested
schema. A small model free-generating raw JSON in prose is unreliable;
constraining it at the token level isn't.

Deliberately NOT one of the 0.5B configs being evaluated — judging a model
with itself (or a sibling config of itself) would be circular.

CAVEAT: even a 4B instruct model is a substantially weaker judge than
Claude. Results from this backend are indicative, not as trustworthy as
what the report's methodology originally called for (Claude-as-judge) — say
so explicitly if these numbers make it into the report.
"""

import json
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as cfg

# transformers>=5 moved PreTrainedTokenizerBase out of tokenization_utils;
# lm-format-enforcer 0.11.3 still imports it from there (same shim as
# backend/server.py).
import transformers.tokenization_utils as _tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PreTrainedTokenizerBase

if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
    _tokenization_utils.PreTrainedTokenizerBase = _PreTrainedTokenizerBase

from lmformatenforcer import JsonSchemaParser  # noqa: E402
from lmformatenforcer.integrations.transformers import (  # noqa: E402
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)

_model = None
_tokenizer = None
_tokenizer_data = None
_lock = threading.Lock()


def _ensure_loaded():
    global _model, _tokenizer, _tokenizer_data
    if _model is not None:
        return
    print(f"[local_judge] loading {cfg.LOCAL_JUDGE_MODEL_ID} (first call only, cached after)...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.LOCAL_JUDGE_MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        cfg.LOCAL_JUDGE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    _model.eval()
    _tokenizer_data = build_token_enforcer_tokenizer_data(_tokenizer)


def call_schema(system: str, user_text: str, schema: dict, max_tokens: int = None) -> dict:
    """Same role as judge_client.call_tool, but takes a bare JSON schema
    (a tool's `input_schema`) instead of a full tool spec, and forces valid
    output via constrained decoding instead of asking nicely.

    Rarely (near a string field's maxLength boundary), lm-format-enforcer
    forces a degenerate token — observed once as a literal U+FFFD
    replacement character — leaving the JSON truncated with an unterminated
    string. The first attempt is greedy (deterministic), so retrying with
    identical inputs would just reproduce the same broken output; retries
    switch to sampling so they actually take a different token path.
    """
    _ensure_loaded()
    max_tokens = cfg.LOCAL_JUDGE_MAX_NEW_TOKENS if max_tokens is None else max_tokens

    inputs = _tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system + "\n\nRespond ONLY with a JSON object matching the required schema."},
            {"role": "user", "content": user_text},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(_tokenizer_data, JsonSchemaParser(schema))

    last_error = None
    last_text = ""
    for attempt in range(cfg.LOCAL_JUDGE_MAX_RETRIES):
        do_sample = attempt > 0
        with _lock, torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=0.8 if do_sample else 1.0,
                pad_token_id=_tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            last_error = error
            last_text = text
            continue
    raise RuntimeError(
        f"local judge produced invalid JSON after {cfg.LOCAL_JUDGE_MAX_RETRIES} attempts: "
        f"{last_error}; last output: {last_text!r}"
    )
