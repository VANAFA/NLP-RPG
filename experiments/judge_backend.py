"""Single entry point judge.py / run_experiment2.py call through, so they
don't need to know whether config.JUDGE_BACKEND is "anthropic" (Claude,
needs ANTHROPIC_API_KEY) or "local" (no key, weaker judge — see
local_judge.py). Both backends are handed the same tool spec (name +
input_schema); "local" just ignores `name` since constrained decoding only
needs the schema itself.
"""

import config as cfg


def call_judge(system: str, user_text: str, tool: dict, max_tokens: int = 300) -> dict:
    if cfg.JUDGE_BACKEND == "local":
        from local_judge import call_schema
        return call_schema(system, user_text, tool["input_schema"], max_tokens)

    from judge_client import call_tool
    return call_tool(system, user_text, tool, max_tokens)
