"""
inference.py — OpenEnv Hackathon submission entry point
========================================================
This file is **required** at the repo root by the evaluation harness.

It exposes two public symbols:
  • predict(obs_dict) -> dict          — single-step inference (observation → action)
  • run_episode(env, seed=None) -> float — full episode loop used by the grader

The agent uses a rule-based heuristic (no API key required) so that the
environment can be validated offline.  Swap in an LLM call inside
`predict()` if you want model-driven decisions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the local email_triage_env package is importable whether this file
# is run from the repo root or from inside the email_triage_env directory.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "email_triage_env"))

from email_triage_env import make                          # noqa: E402
from email_triage_env.models import Action, ActionType, Observation  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: LLM-backed inference via OpenRouter (free tier)
# Set OPENROUTER_API_KEY to enable; falls back to heuristic if unset.
# ---------------------------------------------------------------------------
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


def _build_llm_client():
    """Return an openai.OpenAI client pointed at OpenRouter, or None."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        return None, None
    base_url = _OPENROUTER_BASE_URL if os.environ.get("OPENROUTER_API_KEY") else None
    client = openai.OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    model = os.environ.get("INFERENCE_MODEL", _DEFAULT_MODEL)
    return client, model


_LLM_CLIENT, _LLM_MODEL = _build_llm_client()


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_prompt(obs: Observation) -> str:
    return f"""You are an AI Email Triage Assistant. Classify the email below with exactly one action.

EMAIL:
- Subject      : {obs.subject}
- Sender       : {obs.sender_email} (tier: {obs.sender_tier.value})
- Urgency      : {obs.urgency_signal:.2f}
- Has attachment: {obs.has_attachment}
- Thread depth : {obs.thread_depth}
- Body snippet : {obs.body_snippet}

ACTIONS (pick one):
  ARCHIVE | LABEL_URGENT | LABEL_DELEGATE | DRAFT_REPLY | ESCALATE | FLAG_SPAM | SNOOZE | NO_OP

For LABEL_DELEGATE add: PAYLOAD: {{"delegate_to": "<team>"}}
For DRAFT_REPLY add:    PAYLOAD: {{"reply_body": "<text>"}}

Respond ONLY in this format:
ACTION: <ActionName>
PAYLOAD: {{...}}   (omit if not needed)
"""


def _parse_llm_response(text: str) -> Action:
    action_match = re.search(r"ACTION:\s*([A-Z_]+)", text, re.IGNORECASE)
    if not action_match:
        raise ValueError("No ACTION line found in LLM response.")
    action_name = action_match.group(1).upper()
    try:
        action_type = ActionType(action_name.lower())
    except ValueError:
        raise ValueError(f"Unknown action: {action_name}")

    payload: dict[str, Any] = {}
    payload_match = re.search(r"PAYLOAD:\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if payload_match:
        try:
            payload = json.loads(payload_match.group(1))
        except json.JSONDecodeError:
            pass

    return Action(action_type=action_type, payload=payload)


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM required)
# ---------------------------------------------------------------------------

def _heuristic_action(obs: Observation) -> Action:
    """Rule-based decision tree — always produces a valid Action."""
    tier = obs.sender_tier.value
    urgency = obs.urgency_signal

    # Obvious spam
    if tier == "spam":
        return Action(action_type=ActionType.FLAG_SPAM, payload={})

    # High-urgency executive/manager → escalate or label urgent
    if urgency >= 0.8 and tier in ("executive", "manager"):
        return Action(action_type=ActionType.ESCALATE, payload={})

    if urgency >= 0.6:
        return Action(action_type=ActionType.LABEL_URGENT, payload={})

    # External senders with attachments → delegate to review team
    if tier == "external" and obs.has_attachment:
        return Action(action_type=ActionType.LABEL_DELEGATE, payload={"delegate_to": "review"})

    # Deep threads with low urgency → snooze
    if obs.thread_depth >= 3 and urgency < 0.3:
        return Action(action_type=ActionType.SNOOZE, payload={})

    # Low-urgency external → archive
    if tier == "external" and urgency < 0.3:
        return Action(action_type=ActionType.ARCHIVE, payload={})

    # Default: no-op (let the environment move forward)
    return Action(action_type=ActionType.NO_OP, payload={})


# ---------------------------------------------------------------------------
# Public API — called by the evaluation harness
# ---------------------------------------------------------------------------

def predict(obs_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Single-step inference.

    Parameters
    ----------
    obs_dict : dict
        A raw observation dictionary as returned by env.reset() / env.step().

    Returns
    -------
    dict
        {"action_type": str, "payload": dict}
    """
    obs = Observation(**obs_dict) if not isinstance(obs_dict, Observation) else obs_dict

    # Try LLM if available
    if _LLM_CLIENT is not None:
        try:
            prompt = _format_prompt(obs)
            response = _LLM_CLIENT.chat.completions.create(
                model=_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            reply = response.choices[0].message.content or ""
            action = _parse_llm_response(reply)
            return {"action_type": action.action_type.value, "payload": action.payload}
        except Exception as exc:
            log.warning("LLM inference failed (%s); using heuristic.", exc)

    # Heuristic fallback
    action = _heuristic_action(obs)
    return {"action_type": action.action_type.value, "payload": action.payload}


def run_episode(env=None, task: str = "easy", seed: int = 42) -> float:
    """
    Run a complete episode and return the final episode score.

    Parameters
    ----------
    env : EmailTriageEnv, optional
        Pre-constructed environment.  A new one is created from `task` if None.
    task : str
        Task difficulty: "easy" | "medium" | "hard".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Episode score in [-1.0, 1.0].
    """
    if env is None:
        env = make(task=task)
        actual_task = task
    else:
        actual_task = getattr(env, "task_level", task)

    obs = env.reset(seed=seed)
    done = False
    episode_score = 0.0
    step_count = 0

    # Print START block for structured output
    sys.stdout.write(f"[START] task={actual_task}\n")
    sys.stdout.flush()

    while not done:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        action_dict = predict(obs_dict)

        action = Action(
            action_type=ActionType(action_dict["action_type"]),
            payload=action_dict.get("payload", {}),
        )
        obs, reward, done, info = env.step(action)
        step_count += 1

        # Print STEP block for each step
        sys.stdout.write(f"[STEP] step={step_count} reward={reward:.4f}\n")
        sys.stdout.flush()

        if done:
            episode_score = info.get("episode_score", 0.0)

    # Print END block with final results
    sys.stdout.write(f"[END] task={actual_task} score={episode_score:.4f} steps={step_count}\n")
    sys.stdout.flush()

    return episode_score


# ---------------------------------------------------------------------------
# CLI entry point — for quick local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference.py standalone for local testing.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    scores = []
    for i in range(args.episodes):
        score = run_episode(task=args.task, seed=args.seed + i)
        scores.append(score)

    if scores:
        mean = sum(scores) / len(scores)
        print(f"\nMean score over {args.episodes} episode(s): {mean:.4f}", flush=True)
