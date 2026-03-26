import pytest
from email_triage_env.generator import generate_inbox

def test_generate_inbox_determinism() -> None:
    inbox1 = generate_inbox(35, seed=42)
    inbox2 = generate_inbox(35, seed=42)
    inbox3 = generate_inbox(35, seed=42)
    
    assert inbox1 == inbox2
    assert inbox2 == inbox3
    assert len(inbox1) == 35

def test_generate_inbox_different_seeds() -> None:
    inbox1 = generate_inbox(20, seed=42)
    inbox2 = generate_inbox(20, seed=43)
    assert inbox1 != inbox2
    
def test_all_required_fields_present() -> None:
    inbox = generate_inbox(5, seed=1)
    required_keys = {
        "message_id", "subject", "sender_email", "sender_tier", "body_snippet", 
        "thread_depth", "has_attachment", "received_at", "urgency_signal", 
        "ground_truth_label", "required_response_type"
    }
    for msg in inbox:
        assert required_keys.issubset(msg.keys())

def test_spoofed_emails() -> None:
    inbox_easy = generate_inbox(35, seed=123, task_level="easy")
    spoofed_easy = [m for m in inbox_easy if "Wire Transfer" in m["subject"]]
    assert len(spoofed_easy) == 0

    inbox_medium = generate_inbox(35, seed=123, task_level="medium")
    spoofed_medium = [m for m in inbox_medium if "Wire Transfer" in m["subject"]]
    assert len(spoofed_medium) == 3
