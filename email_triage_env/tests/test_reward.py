from typing import Any
from email_triage_env.models import Action, ActionType, SenderTier
from email_triage_env.reward import compute_reward

def test_all_correct_trajectory() -> None:
    action = Action(action_type=ActionType.LABEL_URGENT, payload={})
    gt = {"ground_truth_label": "urgent", "urgency_signal": 0.9}
    ctx: dict[str, Any] = {}
    r = compute_reward(action, gt, ctx)
    assert abs(r - 0.50) < 1e-5

    action = Action(action_type=ActionType.LABEL_DELEGATE, payload={"delegate_to": "IT"})
    gt = {"ground_truth_label": "delegate", "expected_delegate": "IT"}
    r = compute_reward(action, gt, ctx)
    assert abs(r - 0.45) < 1e-5

    action = Action(action_type=ActionType.FLAG_SPAM, payload={})
    gt = {"ground_truth_label": "spam"}
    r = compute_reward(action, gt, ctx)
    assert abs(r - 0.55) < 1e-5

def test_all_wrong_trajectory() -> None:
    action = Action(action_type=ActionType.FLAG_SPAM, payload={})
    gt = {"ground_truth_label": "archive"}
    ctx: dict[str, Any] = {}
    r = compute_reward(action, gt, ctx)
    assert abs(r - (-0.55)) < 1e-5

    action = Action(action_type=ActionType.ESCALATE, payload={})
    gt = {"ground_truth_label": "archive", "sender_tier": SenderTier.PEER.value, "urgency_signal": 0.1}
    r = compute_reward(action, gt, ctx)
    assert abs(r - (-0.25)) < 1e-5

def test_loop_penalty() -> None:
    action = Action(action_type=ActionType.NO_OP, payload={})
    gt = {"ground_truth_label": "archive"}
    r = compute_reward(action, gt, {"consecutive_no_ops": 2})
    assert abs(r - 0.0) < 1e-5

    r = compute_reward(action, gt, {"consecutive_no_ops": 3})
    assert abs(r - (-0.15)) < 1e-5
    
    r = compute_reward(action, gt, {"consecutive_no_ops": 4})
    assert abs(r - (-0.15)) < 1e-5

def test_budget_overage() -> None:
    action = Action(action_type=ActionType.NO_OP, payload={})
    r = compute_reward(action, {}, {"steps_over_budget": 1})
    assert abs(r - (-0.10)) < 1e-5

def test_policy_violation_termination() -> None:
    action = Action(action_type=ActionType.DRAFT_REPLY, payload={"reply_body": "secret 123"})
    gt = {"ground_truth_label": "reply"}
    ctx: dict[str, Any] = {"task_level": "hard", "contains_confidential": True}
    r = compute_reward(action, gt, ctx)
    assert abs(r - (-0.70)) < 1e-5
    assert ctx.get("done") is True

def test_reply_quality() -> None:
    action = Action(action_type=ActionType.DRAFT_REPLY, payload={"reply_body": "hello"})
    gt = {"ground_truth_label": "reply"}
    ctx: dict[str, Any] = {"task_level": "hard", "llm_grader_score": 0.8}
    r = compute_reward(action, gt, ctx)
    assert abs(r - 0.62) < 1e-5
