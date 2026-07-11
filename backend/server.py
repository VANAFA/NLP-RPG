"""Local inference server for the NLP-RPG narrator LLM.

Loads a Qwen base model (Qwen3-4B-Instruct-2507 by default), optionally
wrapped with the fireball-nlp fireball-qwen3-4b-lora-10k LoRA adapter
and/or activation-steered (see steering.py), and exposes an
OpenAI-compatible `/v1/chat/completions` endpoint matching what
src/llmService.ts expects (the same shape the Kaggle/vLLM notebook target
uses), so this server can be run locally (optionally behind a tunnel) as a
drop-in replacement: just paste this server's URL into the app's settings
(gear icon) panel.

Which model/adapter/steering combination loads is controlled by the
NARRATOR_MODEL_CONFIG env var (see MODEL_CONFIGS below and the root
README's "Backend" section). It defaults to "base_steered" — Base +
Activation Steering — per the recommendation in
experiments/paper/noob_paper_en.tex: among the four configs compared
there, steering the base model reached the highest thematic adherence at
roughly half the latency cost of steering the FIREBALL-finetuned model.

Responses are schema-constrained via lm-format-enforcer: fireball-tuned
models don't reliably emit valid tool-call JSON on their own, so generation
is forced at the token level to match the requested `response_format` JSON
schema (or a sane default), guaranteeing well-formed output every time.
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

import caveman
import steering

# transformers>=5 moved PreTrainedTokenizerBase out of tokenization_utils;
# lm-format-enforcer 0.11.3 still imports it from there. Shim it back in
# before importing lmformatenforcer so its import succeeds.
import transformers.tokenization_utils as _tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PreTrainedTokenizerBase

if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
    _tokenization_utils.PreTrainedTokenizerBase = _PreTrainedTokenizerBase

from lmformatenforcer import JsonSchemaParser  # noqa: E402
from lmformatenforcer.integrations.transformers import (  # noqa: E402
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)

# --- Model configuration ---------------------------------------------------
#
# Mirrors experiments/config.py's MODEL_CONFIGS: only two underlying weight
# sets ever load (base vs. LoRA-adapted) — "steered" just toggles a forward
# hook (steering.py) around generation rather than needing separate models.
MODEL_CONFIGS = {
    "base": {"use_lora": False, "use_steering": False},
    "base_steered": {"use_lora": False, "use_steering": True},
    "finetuned": {"use_lora": True, "use_steering": False},
    "finetuned_steered": {"use_lora": True, "use_steering": True},
}

BASE_MODEL = os.environ.get("NARRATOR_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ADAPTER = os.environ.get("NARRATOR_LORA_ADAPTER", "fireball-nlp/fireball-qwen3-4b-lora-10k")
MODEL_CONFIG_KEY = os.environ.get("NARRATOR_MODEL_CONFIG", "base_steered")
if MODEL_CONFIG_KEY not in MODEL_CONFIGS:
    raise ValueError(
        f"Unknown NARRATOR_MODEL_CONFIG {MODEL_CONFIG_KEY!r}; expected one of {list(MODEL_CONFIGS)}"
    )
USE_LORA = MODEL_CONFIGS[MODEL_CONFIG_KEY]["use_lora"]
USE_STEERING = MODEL_CONFIGS[MODEL_CONFIG_KEY]["use_steering"]

STEERING_LAYER_FRACTION = float(os.environ.get("NARRATOR_STEERING_LAYER_FRACTION", "0.5"))
STEERING_ALPHA = float(os.environ.get("NARRATOR_STEERING_ALPHA", "8.0"))
CONTRASTIVE_PAIRS_PATH = Path(__file__).resolve().parent / "data" / "contrastive_pairs.json"

# CAVEMAN history compression level applied to every message except the
# latest one (see compress_history below). Per the same paper: Medium is
# the best-understood middle ground (~36% tokens saved, ~0.52 retention) —
# Light currently strips almost nothing, and Aggressive collapses factual
# retention (~0.34) — so it's the default rather than "none".
COMPRESSION_LEVEL = os.environ.get("NARRATOR_COMPRESSION_LEVEL", "medium")
if COMPRESSION_LEVEL not in ("none", *caveman.LEVELS.keys()):
    raise ValueError(
        f"Unknown NARRATOR_COMPRESSION_LEVEL {COMPRESSION_LEVEL!r}; "
        f"expected one of {['none', *caveman.LEVELS.keys()]}"
    )

MODEL_NAME = f"narrator-{MODEL_CONFIG_KEY}"
MAX_TOKENS_CAP = 2048

app = FastAPI(title="NLP-RPG Narrator LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(_base_model, ADAPTER) if USE_LORA else _base_model
model.eval()

# Analyzing the tokenizer's vocabulary is expensive and schema-independent,
# so it's done once at startup rather than on every request (per
# lm-format-enforcer's own API for this: passing a raw tokenizer to
# build_transformers_prefix_allowed_tokens_fn silently re-does this work
# every call, which is why unoptimized constrained decoding was ~15x
# slower than normal generation here).
_tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

# transformers .generate() is not safe to call concurrently on one model
# instance; the RPG only has one player at a time so a simple lock is enough.
generate_lock = threading.Lock()

if USE_STEERING:
    with open(CONTRASTIVE_PAIRS_PATH, "r", encoding="utf-8") as _f:
        _contrastive_pairs = json.load(_f)
    STEERING_LAYER_IDX = steering.pick_layer_index(model, STEERING_LAYER_FRACTION)
    STEERING_VECTOR = steering.compute_steering_vector(
        model, tokenizer, STEERING_LAYER_IDX, _contrastive_pairs
    )
else:
    STEERING_LAYER_IDX = 0
    STEERING_VECTOR = None

print(
    f"[narrator] model_config={MODEL_CONFIG_KEY} base={BASE_MODEL} "
    f"lora={ADAPTER if USE_LORA else None} steering={USE_STEERING} "
    f"(layer={STEERING_LAYER_IDX}, alpha={STEERING_ALPHA}) "
    f"compression={COMPRESSION_LEVEL}"
)


# --- Default (turn) response schema -----------------------------------
#
# 'move' and 'spawnNpc' are intentionally absent — the narrator often narrates
# movement/NPCs without remembering to call the matching tool, so those are
# decided by dedicated analyzer calls from the frontend instead (see
# src/llmService.ts's analyzeNarratorTurn).
#
# args is a Union of concrete per-tool models rather than a bare Dict[str,
# Any]: an untyped object schema makes lm-format-enforcer crash as soon as
# the model writes a real key into it (AttributeError: 'bool' object has no
# attribute 'anyOf' — its parser falls back to `additionalProperties`, which
# Pydantic emits as a bare boolean rather than a schema it knows how to walk).

class AddItemArgs(BaseModel):
    name: str
    quantity: int
    kind: str
    desc: str


class DropItemArgs(BaseModel):
    index: int


class SetObjectiveArgs(BaseModel):
    objective: str


class RememberArgs(BaseModel):
    note: str


class ToolCall(BaseModel):
    name: Literal["addItem", "dropItem", "setObjective", "remember"]
    args: Union[AddItemArgs, DropItemArgs, SetObjectiveArgs, RememberArgs]


class NarratorResponse(BaseModel):
    thinking: str
    story: str
    toolCalls: List[ToolCall] = []


DEFAULT_SCHEMA = NarratorResponse.model_json_schema()


# --- Request models ------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None
    schema_: Dict[str, Any] = Field(default_factory=dict, alias="schema")


class ResponseFormat(BaseModel):
    type: str
    json_schema: Optional[JsonSchemaSpec] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    response_format: Optional[ResponseFormat] = None


def compress_history(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Apply CAVEMAN compression (caveman.py) to every user/assistant
    message except the last one, leaving the system prompt and the
    player's current turn untouched. A no-op on today's single-turn frontend
    (system + current user message only — see src/llmService.ts), but this
    server speaks the general OpenAI chat shape, so any caller that
    accumulates multi-turn history gets the same compression the paper
    measured (experiments/caveman.py), applied per-message rather than to
    one joined log."""
    if COMPRESSION_LEVEL == "none" or len(messages) <= 1:
        return messages

    compressed = []
    for m in messages[:-1]:
        if m.role in ("user", "assistant") and m.content.strip():
            compressed.append(ChatMessage(role=m.role, content=caveman.compress_log(m.content, level=COMPRESSION_LEVEL) or m.content))
        else:
            compressed.append(m)
    compressed.append(messages[-1])
    return compressed


def generate_reply(
    messages: List[ChatMessage], temperature: float, max_tokens: int, schema: Dict[str, Any]
) -> str:
    messages = compress_history(messages)
    inputs = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(_tokenizer_data, JsonSchemaParser(schema))

    with torch.no_grad(), steering.SteeringController(model, STEERING_LAYER_IDX, STEERING_VECTOR, STEERING_ALPHA):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, MAX_TOKENS_CAP),
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_fn,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_config": MODEL_CONFIG_KEY,
        "base_model": BASE_MODEL,
        "lora_adapter": ADAPTER if USE_LORA else None,
        "steering": USE_STEERING,
        "compression_level": COMPRESSION_LEVEL,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    schema = DEFAULT_SCHEMA
    if req.response_format and req.response_format.json_schema and req.response_format.json_schema.schema_:
        schema = req.response_format.json_schema.schema_

    with generate_lock:
        content = generate_reply(req.messages, req.temperature, req.max_tokens, schema)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
