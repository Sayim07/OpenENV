import argparse
import json
import logging
import math
import os
import re
import sys

from email_triage_env import make
from email_triage_env.models import Action, ActionType, Observation

try:
    import openai
except ImportError:
    pass

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

# ─── Free models available on OpenRouter (no cost) ────────────────────────────
FREE_MODELS = {
    "llama3":    "meta-llama/llama-3.1-8b-instruct:free",
    "mistral":   "mistralai/mistral-7b-instruct:free",
    "gemma":     "google/gemma-3-4b-it:free",
    "qwen":      "qwen/qwen-2.5-7b-instruct:free",
}
DEFAULT_FREE_MODEL = FREE_MODELS["llama3"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def format_prompt(obs: Observation) -> str:
    """
    Formats the observation into a comprehensive zero-shot prompt.
    """
    return f"""You are an AI Email Triage Assistant. Perform zero-shot classification on the current email.

EMAIL OBSERVATION:
- Message ID: {obs.message_id}
- Subject: {obs.subject}
- Sender Email: {obs.sender_email}
- Sender Tier: {obs.sender_tier.value}
- Received At: {obs.received_at.isoformat()}
- Has Attachment: {obs.has_attachment}
- Thread Depth: {obs.thread_depth}
- Urgency Signal (0-1): {obs.urgency_signal:.2f}

BODY SNIPPET:
{obs.body_snippet}

---

AVAILABLE ACTIONS (pick exactly one):
1. ARCHIVE
2. LABEL_URGENT
3. LABEL_DELEGATE (requires payload: delegate_to)
4. DRAFT_REPLY (requires payload: reply_body)
5. ESCALATE
6. FLAG_SPAM
7. SNOOZE
8. NO_OP

Respond STRICTLY with the action name, optionally followed by payload parameters inside valid JSON brackets.
Format must be exactly:
ACTION: <ActionName>
PAYLOAD: {{"key": "value"}}  (only if required, otherwise omit entirely or supply empty braces)

Examples:
ACTION: ARCHIVE

ACTION: LABEL_DELEGATE
PAYLOAD: {{"delegate_to": "IT"}}
"""


def parse_response(text: str) -> Action:
    """
    Regex parses strictly compliant action names and payload from the LLM prompt.
    Raises ValueError on parse failure.
    """
    action_match = re.search(r"ACTION:\s*([A-Z_]+)", text, re.IGNORECASE)
    if not action_match:
        raise ValueError("Could not parse ACTION line from response.")

    action_name = action_match.group(1).upper()
    try:
        action_type = ActionType(action_name.lower())
    except ValueError:
        raise ValueError(f"Invalid enumeration action type: {action_name}")

    payload = {}
    payload_match = re.search(r"PAYLOAD:\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if payload_match:
        try:
            payload = json.loads(payload_match.group(1))
        except json.JSONDecodeError:
            pass

    return Action(action_type=action_type, payload=payload)


def run_episode(task: str, model: str, seed: int, client: "openai.OpenAI") -> float:
    """
    Spins up the environment, requests LLM generations sequentially.
    """
    env = make(task)
    obs = env.reset(seed=seed)

    done = False
    episode_score = 0.0
    env._emit_telemetry(active_agent=model)

    while not done:
        prompt = format_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            reply_text = response.choices[0].message.content or ""

            try:
                action = parse_response(reply_text)
            except Exception as parse_err:
                logging.warning(f"Parse failure: {parse_err}. Falling back to NO_OP.")
                action = Action(action_type=ActionType.NO_OP, payload={})

            obs, _, done, info = env.step(action)
            if done:
                episode_score = info.get("episode_score", 0.0)

        except Exception as e:
            logging.error(f"API interaction error during loop: {e}")
            raise e

    return episode_score


def compute_stats(scores: list[float]) -> tuple[float, float]:
    """Calculates standard arithmetic mean and standard deviation."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    mean = sum(scores) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std = math.sqrt(variance)
    return mean, std


def build_client(provider: str, model: str) -> tuple["openai.OpenAI", str]:
    """
    Builds an OpenAI-compatible client for the given provider.
    Returns (client, resolved_model_name).
    """
    try:
        openai  # noqa: F821
    except NameError:
        logging.error("OpenAI package not installed. Run: pip install openai")
        sys.exit(1)

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print(
                "\n❌  OPENROUTER_API_KEY not set.\n"
                "\nGet a FREE key (no credit card needed):\n"
                "  1. Go to  https://openrouter.ai\n"
                "  2. Sign in with Google / GitHub\n"
                "  3. Go to  https://openrouter.ai/keys  → 'Create Key'\n"
                "  4. Copy the key (starts with sk-or-...)\n"
                "\nThen run:\n"
                "  $env:OPENROUTER_API_KEY = 'sk-or-...'\n"
                "  python baseline.py\n",
                file=sys.stderr,
            )
            sys.exit(1)

        # Resolve short alias → full model ID
        resolved = FREE_MODELS.get(model, model)
        client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        return client, resolved

    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "\n❌  OPENAI_API_KEY not set.\n"
                "Set it with:  $env:OPENAI_API_KEY = 'sk-...'\n",
                file=sys.stderr,
            )
            sys.exit(1)
        return openai.OpenAI(api_key=api_key), model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline zero-shot LLM inference agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Provider options:
  openrouter (default, FREE)  — requires OPENROUTER_API_KEY env var
  openai                      — requires OPENAI_API_KEY env var

Free model aliases (--provider openrouter):
  llama3   →  {FREE_MODELS['llama3']}
  mistral  →  {FREE_MODELS['mistral']}
  gemma    →  {FREE_MODELS['gemma']}
  qwen     →  {FREE_MODELS['qwen']}

Examples:
  python baseline.py
  python baseline.py --model mistral --task medium
  python baseline.py --provider openai --model gpt-4o-mini
""",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "openai"],
        default="openrouter",
        help="LLM provider to use (default: openrouter, free tier)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["easy", "medium", "hard"],
        default="easy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Model name or alias (default: llama3 → meta-llama/llama-3.1-8b-instruct:free)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    client, resolved_model = build_client(args.provider, args.model)

    print(f"🚀  Provider : {args.provider}")
    print(f"🤖  Model    : {resolved_model}")
    print(f"📋  Task     : {args.task}")
    print(f"🔁  Episodes : {args.episodes}")
    print()

    scores = []
    for i in range(args.episodes):
        print(f"  ▶ Episode {i + 1}/{args.episodes}  (seed={args.seed + i})")
        try:
            score = run_episode(args.task, resolved_model, args.seed + i, client)
            scores.append(score)
            print(f"    ✓ Score: {score:.4f}")
        except Exception as e:
            logging.error(f"Episode {i + 1} failed: {e}")
            sys.exit(1)

    mean, std = compute_stats(scores)

    result = {
        "provider": args.provider,
        "model": resolved_model,
        "task": args.task,
        "mean_score": round(mean, 4),
        "std": round(std, 4),
        "episodes": args.episodes,
        "scores": [round(s, 4) for s in scores],
    }

    print()
    print("📊  Results:")
    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
