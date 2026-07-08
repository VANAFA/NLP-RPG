"""Local inference server for the NLP-RPG narrator LLM.

Loads the Qwen3-4B-Instruct-2507 base model plus the fireball-nlp
fireball-qwen3-4b-lora-10k LoRA adapter, and exposes it over HTTP so the
Vite/React frontend (or a Cloudflare tunnel in front of it) can call it.
"""

import threading
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "fireball-nlp/fireball-qwen3-4b-lora-10k"
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = (
    "Sos el narrador de un RPG estilo terminal DOS retro. Describis el mundo, "
    "reaccionas a las acciones del jugador y controlas a los NPCs de forma "
    "coherente con el estado de juego (ubicacion, HP, XP, nivel, stats, "
    "inventario y NPC activo) que se te da como contexto. Respondes en "
    "castellano, en tono narrativo breve y directo, sin salirte del personaje."
)

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


class Location(BaseModel):
    x: int
    y: int
    label: str


class InventoryItem(BaseModel):
    name: str
    quantity: int
    kind: str


class ActiveNpc(BaseModel):
    name: str
    title: str
    seed: float


class ChatRequest(BaseModel):
    instruction: str
    location: Location
    hp: str
    level: int
    xp: str
    stats: Dict[str, int]
    inventory: List[InventoryItem]
    activeNpc: Optional[ActiveNpc] = None


def build_user_prompt(req: ChatRequest) -> str:
    inventory_desc = (
        ", ".join(f"{item.name} x{item.quantity} ({item.kind})" for item in req.inventory)
        or "vacio"
    )
    stats_desc = ", ".join(f"{name}: {value}" for name, value in req.stats.items())
    npc_desc = (
        f"{req.activeNpc.name} ({req.activeNpc.title})" if req.activeNpc else "ninguno"
    )

    return (
        f"[ESTADO] ubicacion: {req.location.label} ({req.location.x},{req.location.y}) | "
        f"HP: {req.hp} | nivel: {req.level} | XP: {req.xp} | stats: {stats_desc} | "
        f"inventario: {inventory_desc} | NPC activo: {npc_desc}\n"
        f"[JUGADOR] {req.instruction}"
    )


def generate_reply(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    user_prompt = build_user_prompt(req)
    with generate_lock:
        text = generate_reply(user_prompt)
    return {"text": text}
