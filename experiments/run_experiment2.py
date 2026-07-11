"""Experiment 2 - CAVEMAN log compression and memory retention.

Plays a scripted campaign through the Experiment-1-winning model config
(config.EXPERIMENT2_MODEL_KEY) to build a real history log, then at
increasing checkpoints (log lengths) compares closed-question answer
accuracy (IRA) across four context conditions — "none" (raw original log)
plus caveman.py's three compression levels (light/medium/aggressive) — to
trace an actual compression-vs-retention trade-off curve, alongside each
level's token compression ratio (TCR).

Ground truth for grading is NOT hardcoded: the scripted turns only *prompt*
the GM toward stating a fact (e.g. "ask the blacksmith his name") — the
model is free to improvise what it actually says. So right after building
the log, each question's reference answer is extracted from what the model
itself said on its `source_turn`, and grading later compares against that
extracted fact, not an assumption about what the model would say.
"""

import json

import pandas as pd
import torch
from tqdm import tqdm

import caveman
import config as cfg
import models
from judge_backend import call_judge

# `fact` needs maxLength for the same reason as judge.py's VERDICT_TOOL
# `notes` field — an unbounded string under local schema-constrained decoding
# can run past max_new_tokens before closing, producing invalid JSON.
EXTRACT_TOOL = {
    "name": "submit_extracted_fact",
    "description": "Submit the fact extracted from the given text.",
    "input_schema": {
        "type": "object",
        "properties": {"fact": {"type": "string", "maxLength": 100}},
        "required": ["fact"],
    },
}

GRADE_TOOL = {
    "name": "submit_grade",
    "description": "Grade whether an answer captures the reference fact.",
    "input_schema": {
        "type": "object",
        "properties": {"correct": {"type": "boolean"}},
        "required": ["correct"],
    },
}


def extract_reference(turn_text: str, extract_prompt: str):
    system = (
        "You extract a specific fact from a short piece of fantasy RPG narration. "
        "Respond ONLY by calling submit_extracted_fact."
    )
    user_text = f"Text: {turn_text}\n\n{extract_prompt}"
    result = call_judge(system, user_text, EXTRACT_TOOL, max_tokens=100)
    fact = (result.get("fact") or "").strip()
    if not fact or fact.upper() == "UNKNOWN":
        return None
    return fact


def grade_answer(question: str, reference: str, answer: str) -> bool:
    system = (
        "You grade whether an answer captures the key reference fact, even if phrased "
        "differently, less precisely, or embedded in more narration. Ignore prose style; "
        "only check factual correctness against the reference. Respond ONLY by calling submit_grade."
    )
    user_text = f"Question: {question}\nReference fact: {reference}\nGiven answer: {answer or '(empty answer)'}"
    result = call_judge(system, user_text, GRADE_TOOL, max_tokens=50)
    return bool(result.get("correct", False))


def build_campaign_log(loaded, turns, use_steering: bool):
    log_pairs = []
    for turn_text in tqdm(turns, desc="building campaign log"):
        messages = [{"role": "system", "content": cfg.GM_SYSTEM_PROMPT}]
        for player_text, gm_text in log_pairs:
            messages.append({"role": "user", "content": player_text})
            messages.append({"role": "assistant", "content": gm_text})
        messages.append({"role": "user", "content": turn_text})

        gm_text, _latency_ms, _hit_max = models.generate_chat(
            loaded,
            messages,
            use_steering=use_steering,
            max_new_tokens=cfg.EXPERIMENT2_MAX_NEW_TOKENS,
            temperature=cfg.EXPERIMENT2_TEMPERATURE,
        )
        log_pairs.append((turn_text, gm_text))
    return log_pairs


def log_to_text(log_pairs, up_to_turn: int) -> str:
    lines = []
    for player_text, gm_text in log_pairs[:up_to_turn]:
        lines.append(f"Player: {player_text}")
        lines.append(f"GM: {gm_text}")
    return "\n".join(lines)


def answer_question(loaded, use_steering: bool, context_text: str, question: str) -> str:
    system = (
        "You answer questions about a role-playing campaign log using only the log's own "
        "content below. Answer in one short sentence. If the log doesn't say, answer \"unknown\"."
    )
    user_text = f"Campaign log so far:\n{context_text}\n\nQuestion: {question}"
    text, _latency_ms, _hit_max = models.generate_chat(
        loaded,
        [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        use_steering=use_steering,
        max_new_tokens=cfg.EXPERIMENT2_MAX_NEW_TOKENS,
        temperature=cfg.EXPERIMENT2_TEMPERATURE,
    )
    return text


def main():
    torch.manual_seed(cfg.SEED)

    with open(cfg.CAMPAIGN_SCRIPT_PATH, "r", encoding="utf-8") as f:
        campaign = json.load(f)
    turns = campaign["turns"]
    questions = campaign["questions"]

    model_cfg = next(c for c in cfg.MODEL_CONFIGS if c["key"] == cfg.EXPERIMENT2_MODEL_KEY)
    print(f"Using Experiment 1 config as the Experiment 2 model: {model_cfg['label']}")
    loaded = models.load_finetuned() if model_cfg["use_lora"] else models.load_base()
    use_steering = model_cfg["use_steering"]

    print(f"Playing through the {len(turns)}-turn scripted campaign to build the history log...")
    log_pairs = build_campaign_log(loaded, turns, use_steering)

    # Building the log holds activations for the longest (final-turn) context
    # this run will ever process; the extraction step below loads a second,
    # larger model (the local judge) on top, so free whatever's reclaimable
    # first rather than let peak VRAM use compound across both.
    torch.cuda.empty_cache()

    print("Extracting ground-truth facts from the model's own generated turns...")
    references = {}
    for q in questions:
        turn_text = log_pairs[q["source_turn"] - 1][1]
        fact = extract_reference(turn_text, q["extract_prompt"])
        status = "ok" if fact is not None else "SKIP (fact not clearly stated)"
        print(f"  [{status}] {q['id']}: {fact!r}")
        references[q["id"]] = fact

    usable_questions = [q for q in questions if references[q["id"]] is not None]
    if not usable_questions:
        cfg.RESULTS_DIR.mkdir(exist_ok=True)
        with open(cfg.EXPERIMENT2_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"model_key": model_cfg["key"], "log": [{"player": p, "gm": g} for p, g in log_pairs], "extracted_references": references},
                f, indent=2, ensure_ascii=False,
            )
        raise RuntimeError(
            "None of the scripted campaign turns produced an extractable fact - the model likely "
            f"didn't follow the prompts as expected. Inspect {cfg.EXPERIMENT2_LOG_PATH} (written "
            "above) and either adjust data/campaign_script.json or re-run (sampling is stochastic)."
        )

    levels = ["none", *caveman.LEVELS.keys()]
    rows = []
    for checkpoint in cfg.EXPERIMENT2_CHECKPOINTS:
        original_text = log_to_text(log_pairs, checkpoint)
        tokens_original = len(loaded.tokenizer(original_text)["input_ids"])
        applicable = [q for q in usable_questions if q["source_turn"] <= checkpoint]
        n_q = len(applicable)

        for level in levels:
            context_text = original_text if level == "none" else caveman.compress_log(original_text, level=level)
            tokens_level = len(loaded.tokenizer(context_text)["input_ids"]) if context_text else 0
            tcr = round(1 - (tokens_level / tokens_original), 3) if tokens_original else float("nan")

            correct = 0
            for q in tqdm(applicable, desc=f"checkpoint {checkpoint} / {level}"):
                answer = answer_question(loaded, use_steering, context_text, q["question"])
                correct += grade_answer(q["question"], references[q["id"]], answer)

            rows.append({
                "checkpoint_turns": checkpoint,
                "level": level,
                "n_questions": n_q,
                "tokens": tokens_level,
                "tcr": tcr,
                "ira": round(correct / n_q, 3) if n_q else float("nan"),
            })

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with open(cfg.EXPERIMENT2_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"model_key": model_cfg["key"], "log": [{"player": p, "gm": g} for p, g in log_pairs], "extracted_references": references},
            f, indent=2, ensure_ascii=False,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.EXPERIMENT2_SUMMARY_CSV_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {cfg.EXPERIMENT2_SUMMARY_CSV_PATH}")
    print(f"Wrote {cfg.EXPERIMENT2_LOG_PATH}")


if __name__ == "__main__":
    main()
