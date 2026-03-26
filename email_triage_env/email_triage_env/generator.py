import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from email_triage_env.models import SenderTier

# Tiers weights
TIERS = [
    SenderTier.EXECUTIVE,
    SenderTier.MANAGER,
    SenderTier.PEER,
    SenderTier.EXTERNAL,
    SenderTier.SPAM,
]
TIER_WEIGHTS = [0.20, 0.25, 0.25, 0.15, 0.15]

RESPONSE_TYPES = ["archive", "urgent", "delegate", "spam", "reply"]

SUBJECTS: dict[SenderTier, list[str]] = {
    SenderTier.EXECUTIVE: ["Q3 Board Deck", "Urgent: Earnings Call", "Strategy Review"],
    SenderTier.MANAGER: ["Weekly 1:1", "Project Update", "Team Offsite", "Approvals needed"],
    SenderTier.PEER: ["Lunch today?", "Code Review", "Help with deployment", "Draft docs"],
    SenderTier.EXTERNAL: ["Vendor Contract", "Partnership Opportunity", "Invoice #1024"],
    SenderTier.SPAM: ["You won a gift card!", "SEO Services", "Earn money fast"],
}

DOMAINS: dict[SenderTier, list[str]] = {
    SenderTier.EXECUTIVE: ["company.com"],
    SenderTier.MANAGER: ["company.com"],
    SenderTier.PEER: ["company.com"],
    SenderTier.EXTERNAL: ["vendor.com", "partner.io", "client.net"],
    SenderTier.SPAM: ["cheap-seo.xyz", "lottery-winner.biz", "free-money.info"],
}


def generate_synthetic_email(tier: SenderTier, rnd: random.Random) -> tuple[str, str, str]:
    """Generates synthetic sender email, subject, and body based on the sender tier."""
    domain = rnd.choice(DOMAINS[tier])

    if tier in (SenderTier.EXECUTIVE, SenderTier.MANAGER, SenderTier.PEER):
        name = rnd.choice(["alex", "jordan", "taylor", "morgan", "casey"])
    elif tier == SenderTier.EXTERNAL:
        name = rnd.choice(["info", "sales", "support", "billing"])
    else:
        name = rnd.choice(["admin", "noreply", "deals"])

    sender_email = f"{name}@{domain}"
    subject = rnd.choice(SUBJECTS[tier])
    body_snippet = f"This is a simulated body for a {tier.value} email. Please review."
    return sender_email, subject, body_snippet


def get_urgency(tier: SenderTier, rnd: random.Random) -> float:
    """Returns an urgency float [0.0, 1.0] correlated with sender_tier."""
    if tier == SenderTier.EXECUTIVE:
        return rnd.uniform(0.7, 1.0)
    elif tier == SenderTier.MANAGER:
        return rnd.uniform(0.4, 0.9)
    elif tier == SenderTier.SPAM:
        return rnd.uniform(0.0, 0.3)
    else:
        return rnd.uniform(0.1, 0.8)


def _generate_spoofed_executive(rnd: random.Random, base_time: datetime) -> dict[str, Any]:
    """Generates a spoofed executive email with sender_tier=external (Task 2/medium)."""
    name = rnd.choice(["ceo", "president", "founder"])
    domain = rnd.choice(["company-secure.com", "company-update.net", "company-it.com"])
    sender_email = f"{name}@{domain}"

    received_at = base_time + timedelta(
        days=rnd.randint(0, 30),
        hours=rnd.randint(0, 23),
        minutes=rnd.randint(0, 59)
    )

    req_type = "spam"

    return {
        "message_id": str(uuid.UUID(int=rnd.getrandbits(128))),
        "subject": "URGENT: Wire Transfer Required Immediately",
        "sender_email": sender_email,
        "sender_tier": SenderTier.EXTERNAL.value,
        "body_snippet": "Please process this wire transfer immediately to our new vendor.",
        "thread_depth": rnd.randint(0, 5),
        "has_attachment": rnd.random() < 0.30,
        "received_at": received_at,
        "urgency_signal": rnd.uniform(0.8, 1.0),
        "required_response_type": req_type,
        "ground_truth_label": req_type,
    }


def generate_inbox(size: int, seed: Optional[int] = None, task_level: str = "easy") -> list[dict[str, Any]]:
    """
    Generates a procedural inbox of emails dynamically.
    
    Args:
        size: The number of emails to generate.
        seed: The random seed to guarantee determinism.
        task_level: Determines difficulty (adds spoofed emails for 'medium' and 'hard').
        
    Returns:
        A list of dictionaries representing the synthetic emails.
    """
    rnd = random.Random(seed) if seed is not None else random.Random()

    inbox: list[dict[str, Any]] = []
    spoof_count = 3 if task_level in ("medium", "hard") else 0
    actual_size = max(0, size - spoof_count)

    base_time = datetime(2025, 1, 1, 9, 0)

    for _ in range(actual_size):
        tier = rnd.choices(TIERS, weights=TIER_WEIGHTS, k=1)[0]
        sender_email, subject, body_snippet = generate_synthetic_email(tier, rnd)
        urgency = get_urgency(tier, rnd)
        has_attachment = rnd.random() < 0.30
        thread_depth = rnd.randint(0, 5)

        received_at = base_time + timedelta(
            days=rnd.randint(0, 30),
            hours=rnd.randint(0, 23),
            minutes=rnd.randint(0, 59)
        )

        # Basic label heuristic setup for required response
        if tier == SenderTier.SPAM:
            req_type = "spam"
        elif tier == SenderTier.EXECUTIVE and urgency > 0.8:
            req_type = "urgent"
        elif tier == SenderTier.MANAGER and thread_depth > 0:
            req_type = "reply"
        elif tier == SenderTier.PEER:
            req_type = rnd.choice(["reply", "delegate"])
        else:
            req_type = rnd.choice(RESPONSE_TYPES)

        msg = {
            "message_id": str(uuid.UUID(int=rnd.getrandbits(128))),
            "subject": subject,
            "sender_email": sender_email,
            "sender_tier": tier.value,
            "body_snippet": body_snippet,
            "thread_depth": thread_depth,
            "has_attachment": has_attachment,
            "received_at": received_at,
            "urgency_signal": urgency,
            "required_response_type": req_type,
            "ground_truth_label": req_type,
        }
        inbox.append(msg)

    # Insert exactly the required spoofed executive emails for task completion checking
    for _ in range(min(spoof_count, size)):
        inbox.append(_generate_spoofed_executive(rnd, base_time))

    rnd.shuffle(inbox)
    return inbox
