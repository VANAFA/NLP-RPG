"""Local inference server for the NLP-RPG narrator LLM.

Loads the Qwen3-4B-Instruct-2507 base model plus the fireball-nlp
fireball-qwen3-4b-lora-10k LoRA adapter, and exposes an OpenAI-compatible
`/v1/chat/completions` endpoint matching what src/llmService.ts expects
(the same shape the Kaggle/vLLM notebook target uses), so this server can
be run locally (optionally behind a tunnel) as a drop-in replacement: just
paste this server's URL into the app's settings (gear icon) panel.

Responses are schema-constrained via lm-format-enforcer: the fireball LoRA
doesn't reliably emit valid tool-call JSON on its own, so generation is
forced at the token level to match the requested `response_format` JSON
schema (or a sane default), guaranteeing well-formed output every time.
"""

import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

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

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "fireball-nlp/fireball-qwen3-4b-lora-10k"
MODEL_NAME = "fireball-qwen3-4b-lora-10k"
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
model = PeftModel.from_pretrained(_base_model, ADAPTER)
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


# --- Default (turn) response schema -----------------------------------

class ToolCall(BaseModel):
    name: Literal["move", "addItem", "dropItem", "spawnNpc", "setObjective", "remember"]
    args: Dict[str, Any] = {}


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


def generate_reply(
    messages: List[ChatMessage], temperature: float, max_tokens: int, schema: Dict[str, Any]
) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(_tokenizer_data, JsonSchemaParser(schema))

    with torch.no_grad():
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
    return {"status": "ok"}


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
