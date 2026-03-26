from email_triage_env.grader import compute_final_score, grade_easy, grade_medium, grade_hard
from typing import Any

def test_compute_final_score() -> None:
    score = compute_final_score(1.0, 1.0, 1.0, 1.0, 0)
    assert score == 1.0
    score_penalty = compute_final_score(1.0, 1.0, 1.0, 1.0, 2)
    assert abs(score_penalty - 0.80) < 1e-5

def test_grade_easy_empty() -> None:
    score = grade_easy([])
    assert abs(score - 0.45) < 1e-5

def test_grade_easy_exhausted() -> None:
    trajectory: list[dict[str, Any]] = [{"action_type": "archive", "ground_truth_label": "archive"} for _ in range(35)]
    score = grade_easy(trajectory)
    assert score < 1.0

def test_grade_easy_all_correct() -> None:
    trajectory: list[dict[str, Any]] = []
    for _ in range(8):
        trajectory.append({"action_type": "archive", "ground_truth_label": "archive"})
    for _ in range(4):
        trajectory.append({"action_type": "flag_spam", "ground_truth_label": "spam"})
    for _ in range(4):
        trajectory.append({"action_type": "label_urgent", "sender_tier": "executive"})
    for _ in range(4):
        trajectory.append({"action_type": "draft_reply", "ground_truth_label": "reply"})
    assert grade_easy(trajectory) == 1.0

def test_grade_easy_all_wrong() -> None:
    trajectory: list[dict[str, Any]] = [{"action_type": "no_op", "ground_truth_label": "urgent"} for _ in range(35)]
    score = grade_easy(trajectory)
    assert abs(score - 0.325) < 1e-5

def test_grade_easy_fp_spam() -> None:
    # Trigger fp_spam penalty
    trajectory: list[dict[str, Any]] = [
        {"action_type": "flag_spam", "ground_truth_label": "archive"}
    ]
    score = grade_easy(trajectory)
    assert score < 0.5

def test_grade_medium_all_correct() -> None:
    trajectory: list[dict[str, Any]] = []
    for i in range(8, 11):
        trajectory.append({"action_type": "label_urgent", "message_id": i, "urgency_signal": 0.9})
    for i in range(3, 8):
        trajectory.append({"action_type": "label_delegate", "expected_delegate": "HR", "payload": {"delegate_to": "HR"}, "message_id": i, "urgency_signal": 0.5})
    for i in range(3):
        trajectory.append({"action_type": "flag_spam", "sender_tier": "external", "ground_truth_label": "spam", "subject": "Wire Transfer", "message_id": i, "urgency_signal": 0.1})
    assert grade_medium(trajectory) == 1.0
    
def test_grade_medium_penalties() -> None:
    # Trigger fp_escalate and max snooze/budget penalties
    trajectory: list[dict[str, Any]] = []
    for i in range(50): # Over 45 steps
        trajectory.append({"action_type": "snooze", "message_id": i, "urgency_signal": 0.1})
    trajectory.append({"action_type": "escalate", "sender_tier": "peer", "message_id": 99})
    score = grade_medium(trajectory)
    assert score < 0.5

def test_grade_hard_urgent_replies() -> None:
    # Urgent messages with replies in time
    trajectory: list[dict[str, Any]] = [
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "draft_reply", "message_id": "u1", "ground_truth_label": "urgent", "llm_grader_score": 0.8}
    ]
    score = grade_hard(trajectory)
    assert score > 0.5

def test_grade_hard_urgent_missed() -> None:
    # Urgent messages with NO reply in time
    trajectory: list[dict[str, Any]] = [
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "no_op", "message_id": "u1", "ground_truth_label": "urgent"},
        {"action_type": "flag_spam", "message_id": "s1", "ground_truth_label": "archive"} # Trigger fp_spam in hard
    ]
    score = grade_hard(trajectory)
    assert score < 0.5

def test_grade_hard_policy_violation() -> None:
    trajectory: list[dict[str, Any]] = []
    for i in range(5):
        trajectory.append({"action_type": "draft_reply", "contains_confidential": True, "message_id": i, "ground_truth_label": "archive"})
    score = grade_hard(trajectory)
    assert score < 0.20

def test_grade_hard_all_wrong() -> None:
    trajectory: list[dict[str, Any]] = []
    for i in range(65):
        trajectory.append({"action_type": "draft_reply", "contains_confidential": True, "message_id": i, "ground_truth_label": "archive"})
    score = grade_hard(trajectory)
    assert score == 0.0
