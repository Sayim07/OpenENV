import pytest
from datetime import datetime
from email_triage_env.models import Action, ActionType, Observation, SenderTier

def test_valid_observation() -> None:
    obs = Observation(
        message_id="123",
        subject="Test",
        sender_email="test@company.com",
        sender_tier=SenderTier.EXTERNAL,
        body_snippet="Hello",
        thread_depth=0,
        has_attachment=False,
        received_at=datetime.now(),
        urgency_signal=0.5,
        inbox_remaining=10,
        step_budget_remaining=20,
        context={}
    )
    assert obs.message_id == "123"

def test_field_validators_subject_urgency() -> None:
    with pytest.raises(ValueError):
        Observation(
            message_id="123",
            subject="A" * 121, # > 120 chars
            sender_email="test@company.com",
            sender_tier=SenderTier.EXTERNAL,
            body_snippet="Hello",
            thread_depth=0,
            has_attachment=False,
            received_at=datetime.now(),
            urgency_signal=0.5,
            inbox_remaining=10,
            step_budget_remaining=20
        )
        
    with pytest.raises(ValueError):
        Observation(
            message_id="123",
            subject="Test",
            sender_email="test@company.com",
            sender_tier=SenderTier.EXTERNAL,
            body_snippet="Hello",
            thread_depth=0,
            has_attachment=False,
            received_at=datetime.now(),
            urgency_signal=1.5, # > 1.0
            inbox_remaining=10,
            step_budget_remaining=20
        )

def test_body_snippet_validator() -> None:
    with pytest.raises(ValueError):
        Observation(
            message_id="123", subject="Test", sender_email="test@company.com",
            sender_tier=SenderTier.EXTERNAL, body_snippet="A" * 501, # > 500 chars
            thread_depth=0, has_attachment=False, received_at=datetime.now(),
            urgency_signal=0.5, inbox_remaining=10, step_budget_remaining=20
        )

def test_non_negative_validators() -> None:
    # thread_depth
    with pytest.raises(ValueError):
        Observation(
            message_id="123", subject="Test", sender_email="test@company.com",
            sender_tier=SenderTier.EXTERNAL, body_snippet="Hello",
            thread_depth=-1, # < 0
            has_attachment=False, received_at=datetime.now(),
            urgency_signal=0.5, inbox_remaining=10, step_budget_remaining=20
        )
    # inbox_remaining
    with pytest.raises(ValueError):
        Observation(
            message_id="123", subject="Test", sender_email="test@company.com",
            sender_tier=SenderTier.EXTERNAL, body_snippet="Hello",
            thread_depth=0, has_attachment=False, received_at=datetime.now(),
            urgency_signal=0.5, inbox_remaining=-1, # < 0
            step_budget_remaining=20
        )

def test_label_delegate_without_payload() -> None:
    with pytest.raises(ValueError):
        Action(action_type=ActionType.LABEL_DELEGATE, payload={})

def test_draft_reply_without_payload() -> None:
    with pytest.raises(ValueError):
        Action(action_type=ActionType.DRAFT_REPLY, payload={})

def test_invalid_action_type_enum() -> None:
    with pytest.raises(ValueError):
        Action(action_type="random_action", payload={}) # type: ignore
