"""Local inference server for the NLP-RPG narrator LLM.

Loads the Qwen3-4B-Instruct-2507 base model plus the fireball-nlp
fireball-qwen3-4b-lora-10k LoRA adapter, and exposes an OpenAI-compatible
`/v1/chat/completions` endpoint matching what src/llmService.ts expects
(the same shape the Kaggle/vLLM notebook target uses), so this server can
be run locally (optionally behind a tunnel) as a drop-in replacement: just
paste this server's URL into the app's settings (gear icon) panel.
"""

import json
import re
import threading
import time
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# transformers .generate() is not safe to call concurrently on one model
# instance; the RPG only has one player at a time so a simple lock is enough.
generate_lock = threading.Lock()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512


def generate_reply(messages: List[ChatMessage], temperature: float, max_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, MAX_TOKENS_CAP),
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def normalize_content(text: str) -> str:
    """Ensure the response is always the {thinking, story, toolCalls} JSON
    contract llmService.ts expects, even when the model answers in plain
    narrative text instead of structured JSON (it often does)."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "story" in parsed:
            return stripped
    except json.JSONDecodeError:
        pass

    return json.dumps({"thinking": "", "story": stripped, "toolCalls": []}, ensure_ascii=False)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    with generate_lock:
        text = generate_reply(req.messages, req.temperature, req.max_tokens)

    content = normalize_content(text)

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
