"""Shared Anthropic client + forced-tool-call helper, used by judge.py
(Experiment 1) and run_experiment2.py (fact extraction + answer grading).
Tool-use with a forced tool_choice is used instead of asking for raw JSON
in prose — much more reliable to parse.
"""

import os
import time

import anthropic

import config as cfg  # also loads experiments/.env as a side effect

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy experiments/.env.example to experiments/.env "
                "and fill in your key, or export it in your shell."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_tool(system: str, user_text: str, tool: dict, max_tokens: int = 300) -> dict:
    """Calls Claude with exactly one forced tool, retrying transient errors
    with backoff. Returns the tool call's `input` dict."""
    last_error = None
    for attempt in range(cfg.JUDGE_MAX_RETRIES):
        try:
            message = get_client().messages.create(
                model=cfg.JUDGE_MODEL,
                max_tokens=max_tokens,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
                messages=[{"role": "user", "content": user_text}],
            )
            for block in message.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    return block.input
            raise RuntimeError(f"Response had no {tool['name']} tool call")
        except Exception as error:  # noqa: BLE001 - retry any transient API error
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Claude call failed after {cfg.JUDGE_MAX_RETRIES} attempts: {last_error}")
