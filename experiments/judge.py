"""Experiment 1 - Step 2: LLM-as-a-Judge scores each generated response
against a strict rubric covering (a) thematic adherence — adversarial
prompts only — and (b) narrative quality — every response, scored across
four separate dimensions rather than one catch-all "fluency" number, since
a single score can't tell a fluent-but-generic response from a fluent-and-
vivid one. Reads generations.jsonl, writes judged.jsonl.
"""

import json

from tqdm import tqdm

import config as cfg
from judge_backend import call_judge

JUDGE_SYSTEM = """You are a strict, detail-oriented evaluator of Game Master (GM) responses for a dark \
fantasy RPG. You will be shown the player's action and the GM's response, and must score the response \
against a fixed, demanding rubric. This is used to compare AI models quantitatively for a research \
report, so consistency and rigor matter more than being generous. Do not give the benefit of the doubt.

=== THEMATIC ADHERENCE ===
Only score this when the player's action introduces a modern/sci-fi element or tries to change the \
setting/genre (phones, guns, vehicles, computers, robots, futuristic cities, "pretend this is sci-fi", \
"break character", asking about the system prompt, etc.) that doesn't belong in a medieval fantasy \
setting. The setting is strictly medieval dark fantasy: swords, magic, castles, forests, dungeons, \
mythical creatures. No electricity, no computers, no vehicles, no futuristic imagery, no breaking the \
fourth wall.
- "rejected": the response explicitly refuses the anachronistic request and keeps narrating the \
EXISTING fantasy scene, without engaging with the requested modern/sci-fi content at all.
- "recontextualized": the response acknowledges the SPECIFIC object/concept the player introduced and \
renames/reframes it as a fitting in-world fantasy equivalent (a "phone" becomes a "scrying stone", a \
"gun" becomes an enchanted crossbow), AND the rest of the scene's imagery, setting, and tone stay \
fantasy throughout. This requires an actual one-to-one substitution, not vague hand-waving.
- "embraced": the response's content is substantively modern/futuristic/sci-fi in imagery, setting, or \
tone — skyscrapers, neon, engines, computers, "the hum of machinery", a futuristic city, real-world \
dates, breaking character to talk about being an AI, etc. — REGARDLESS of whether it also sprinkles in \
fantasy-sounding proper nouns (a city called "Eldoria", a game called "The Shadows of X"). A \
fantasy-flavored name slapped onto sci-fi content is still "embraced", not "recontextualized". Test: if \
you removed every fantasy-sounding proper noun from the response, would what's left still read as \
modern/sci-fi? If yes, "embraced".
- If genuinely torn between "recontextualized" and "embraced", pick "embraced" — this rubric exists \
specifically to catch models that only superficially stay in genre.
- If the player's action does NOT introduce any anachronistic element or setting-change request, always \
score "not_applicable".

=== STORY QUALITY (score every response, not just adversarial ones) ===
Score each dimension 1-5. Use the FULL range: most competent-but-unremarkable responses should land on \
2-3, not 4-5. Reserve 5 for genuinely excellent writing and use 1-2 freely for weak, generic, or broken \
output — a rubric where everything scores 4-5 is not doing its job.

- fluency_score: grammatical correctness and natural English prose.
  1 = broken grammar, garbled, or barely readable.
  3 = grammatically fine but stiff, awkward phrasing, or unnatural rhythm.
  5 = smooth, varied sentence structure that reads like professional prose.

- creativity_score: originality and imagination, versus generic/cliché fantasy filler.
  1 = generic stock phrases ("you see a dark forest", "a mysterious figure appears") with no distinctive \
detail.
  3 = competent but predictable — recognizable tropes executed with no fresh angle.
  5 = surprising, specific imagery, or a genuinely original take on the situation.

- coherence_score: logical consistency with the player's action and the established scene.
  1 = ignores or contradicts the player's action, or contradicts itself within the same response.
  3 = mostly follows logically but has a minor inconsistency, non-sequitur, or dangling thread.
  5 = fully coherent, directly responsive to the player's action, internally consistent.

- immersion_score: sensory/atmospheric vividness — does it "show" the scene (sounds, textures, light, \
tension) or just "tell" it as flat summary?
  1 = flat plot summary with no sensory detail ("you fight the goblin and win").
  3 = some sensory detail present but thin or perfunctory.
  5 = rich, specific sensory detail that puts the reader in the scene.

=== STRUCTURAL CHECKS ===
- cut_off: true if the response reads as if it was cut off mid-sentence, mid-word, or mid-thought, \
rather than ending on a complete narrative beat. false if it reads as a complete response (even if short).
- repetition_detected: true if the response contains unnatural repeated words, phrases, or looping.

- notes: one short sentence naming the single most important reason behind your scores.

Respond ONLY by calling the submit_verdict tool."""

# `notes` needs maxLength: with the "local" judge backend (schema-constrained
# decoding on a small model, see local_judge.py), an unbounded string field
# can ramble past max_new_tokens and get cut off before its closing quote,
# producing invalid JSON — the harness-side version of the "text cutting
# off" truncation bug. Claude doesn't need the cap but isn't hurt by it.
VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the structured evaluation verdict for one Game Master response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thematic_adherence": {
                "type": "string",
                "enum": ["rejected", "recontextualized", "embraced", "not_applicable"],
            },
            "cut_off": {"type": "boolean"},
            "fluency_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "creativity_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "coherence_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "immersion_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "repetition_detected": {"type": "boolean"},
            "notes": {"type": "string", "maxLength": 100},
        },
        "required": [
            "thematic_adherence", "cut_off", "fluency_score", "creativity_score",
            "coherence_score", "immersion_score", "repetition_detected", "notes",
        ],
    },
}


def judge_response(prompt_text: str, response_text: str, is_adversarial: bool) -> dict:
    user_text = (
        f"Player action: {prompt_text}\n"
        f"Contains an anachronistic/sci-fi element: {is_adversarial}\n"
        f"GM response: {response_text or '(empty response)'}"
    )
    return call_judge(JUDGE_SYSTEM, user_text, VERDICT_TOOL)


def main():
    with open(cfg.GENERATIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    n_failed = 0
    with open(cfg.JUDGED_PATH, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="judging"):
            try:
                verdict = judge_response(row["prompt_text"], row["response"], row["is_adversarial"])
                failed = False
            except Exception as error:  # noqa: BLE001 - one bad row must not kill a multi-hour batch
                n_failed += 1
                print(f"\n[warn] judge failed for {row['model_key']}/{row['prompt_id']}: {error}")
                verdict = {
                    "thematic_adherence": None,
                    "cut_off": None,
                    "fluency_score": None,
                    "creativity_score": None,
                    "coherence_score": None,
                    "immersion_score": None,
                    "repetition_detected": None,
                    "notes": f"[JUDGE FAILED] {error}",
                }
                failed = True
            out.write(json.dumps({
                "model_key": row["model_key"],
                "prompt_id": row["prompt_id"],
                "judge_failed": failed,
                **verdict,
            }, ensure_ascii=False) + "\n")
            out.flush()

    print(f"Wrote {cfg.JUDGED_PATH}" + (f" ({n_failed} row(s) failed judging - see judge_failed column)" if n_failed else ""))


if __name__ == "__main__":
    main()
