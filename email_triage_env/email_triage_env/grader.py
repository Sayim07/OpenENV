from typing import Any

def compute_final_score(
    classification_accuracy: float,
    priority_ordering_score: float,
    delegation_quality: float,
    completion_rate: float,
    reply_policy_violations: int = 0
) -> float:
    """
    Computes final grade matching formula precisely in [0.0, 1.0].
    
    score = (0.40 * classification_accuracy)
          + (0.25 * priority_ordering_score)
          + (0.20 * delegation_quality)
          + (0.15 * completion_rate)
          - (0.10 * reply_policy_violations)
    """
    score = (
        0.40 * classification_accuracy +
        0.25 * priority_ordering_score +
        0.20 * delegation_quality +
        0.15 * completion_rate
    )
    score -= (0.10 * reply_policy_violations)
    return max(0.0, min(1.0, score))

def grade_easy(trajectory: list[dict[str, Any]]) -> float:
    """
    Grades the 'easy' task using its designated heuristics.
    """
    archived = 0
    spam_flagged = 0
    fp_spam = 0
    exec_urgent = 0
    max_consecutive_no_ops = 0
    current_no_ops = 0
    processed_count = 0
    
    for t in trajectory:
        action = t.get("action_type")
        gt_label = t.get("ground_truth_label")
        tier = t.get("sender_tier")
        
        if action == "no_op":
            current_no_ops += 1
            max_consecutive_no_ops = max(max_consecutive_no_ops, current_no_ops)
        else:
            current_no_ops = 0
            if action not in ("snooze", "escalate"):
                processed_count += 1
                
        if action == "archive" and gt_label == "archive":
            archived += 1
        elif action == "flag_spam":
            if gt_label == "spam":
                spam_flagged += 1
            else:
                fp_spam += 1
        elif action == "label_urgent" and tier == "executive":
            exec_urgent += 1

    c_archive = min(1.0, archived / 8.0)
    c_spam = min(1.0, spam_flagged / 4.0)
    if fp_spam > 0:
        c_spam = 0.0 # penalty for any false positive
    c_exec = min(1.0, exec_urgent / 4.0)
    
    classification_accuracy = (c_archive + c_spam + c_exec) / 3.0
    
    priority_ordering_score = 1.0 if max_consecutive_no_ops < 3 else 0.5
    
    delegation_quality = 1.0
    
    if len(trajectory) <= 30:
        completion_rate = min(1.0, processed_count / 20.0)
    else:
        completion_rate = 0.5 * min(1.0, processed_count / 20.0)
        
    return compute_final_score(
        classification_accuracy, 
        priority_ordering_score, 
        delegation_quality, 
        completion_rate
    )

def grade_medium(trajectory: list[dict[str, Any]]) -> float:
    """
    Grades the 'medium' task using its designated heuristics.
    """
    spoofed_flagged = 0
    fp_escalate = 0
    correct_delegations = 0
    snooze_count = 0
    
    message_action_steps: dict[Any, dict[str, Any]] = {}
    
    for step_idx, t in enumerate(trajectory):
        action = t.get("action_type")
        gt_label = t.get("ground_truth_label")
        tier = t.get("sender_tier")
        msg_id = t.get("message_id", step_idx)
        urgency = t.get("urgency_signal", 0.0)
        
        if action != "no_op":
            if msg_id not in message_action_steps:
                message_action_steps[msg_id] = {
                    "step": step_idx, 
                    "urgency": urgency,
                    "action": action
                }
                
        is_spoofed = (tier == "external" and gt_label == "spam" and "Wire Transfer" in t.get("subject", ""))
        if is_spoofed and action == "flag_spam":
            spoofed_flagged += 1
            
        if action == "label_delegate":
            expected_role = t.get("expected_delegate")
            chosen_role = t.get("payload", {}).get("delegate_to")
            if expected_role and chosen_role == expected_role:
                correct_delegations += 1
                
        if action == "escalate":
            if tier in ("peer", "external"):
                fp_escalate += 1
                
        if action == "snooze":
            snooze_count += 1

    spoof_score = min(1.0, spoofed_flagged / 3.0)
    escalate_score = 1.0 if fp_escalate == 0 else max(0.0, 1.0 - 0.2 * fp_escalate)
    classification_accuracy = (spoof_score + escalate_score) / 2.0
    
    delegation_quality = min(1.0, correct_delegations / 5.0)
    
    unique_msgs = []
    seen = set()
    for step_idx, t in enumerate(trajectory):
        m_id = t.get("message_id", step_idx)
        if m_id not in seen:
            seen.add(m_id)
            unique_msgs.append({
                "id": m_id,
                "urgency": t.get("urgency_signal", 0.0)
            })
            
    unique_msgs.sort(key=lambda x: x["urgency"], reverse=True)
    top_3_true = [m["id"] for m in unique_msgs[:3]]
    
    acted_msgs_ids = sorted(message_action_steps.keys(), key=lambda k: message_action_steps[k]["step"])
    
    ordering_hits = 0
    for true_rank in range(min(3, len(top_3_true))):
        msg_id = top_3_true[true_rank]
        if msg_id in acted_msgs_ids:
            actual_rank = acted_msgs_ids.index(msg_id)
            if abs(actual_rank - true_rank) <= 1:
                ordering_hits += 1
                
    priority_ordering_score = ordering_hits / 3.0 if len(top_3_true) >= 3 else 1.0
    
    if len(trajectory) <= 45 and snooze_count <= 2:
        completion_rate = 1.0
    else:
        penalty = 0.0
        if len(trajectory) > 45:
            penalty += 0.5
        if snooze_count > 2:
            penalty += 0.5
        completion_rate = max(0.0, 1.0 - penalty)
        
    return compute_final_score(
        classification_accuracy, 
        priority_ordering_score, 
        delegation_quality, 
        completion_rate
    )

def grade_hard(trajectory: list[dict[str, Any]]) -> float:
    """
    Grades the 'hard' task using its designated heuristics and subtracts policy violations.
    """
    urgent_replied_in_time = 0
    llm_scores = []
    
    fp_spam = 0
    total_spam_flags = 0
    policy_violations = 0
    
    processed_msg_ids = set()
    unique_msg_ids = set()
    
    for idx, t in enumerate(trajectory):
        action = t.get("action_type")
        gt_label = t.get("ground_truth_label")
        msg_id = t.get("message_id", idx)
        unique_msg_ids.add(msg_id)
        
        if action != "no_op":
            processed_msg_ids.add(msg_id)
            
        if t.get("contains_confidential") and action == "draft_reply":
            policy_violations += 1
            
        if action == "flag_spam":
            total_spam_flags += 1
            if gt_label != "spam":
                fp_spam += 1
                
        if action == "draft_reply":
            score = t.get("llm_grader_score", 0.0)
            llm_scores.append(score)
            
    urgent_msgs = set(t.get("message_id", i) for i, t in enumerate(trajectory) if t.get("ground_truth_label") == "urgent")
    total_urgent_messages = len(urgent_msgs)
    
    for um_id in urgent_msgs:
        steps_for_um = [i for i, t in enumerate(trajectory) if t.get("message_id", i) == um_id]
        if not steps_for_um:
            continue
        first_seen = steps_for_um[0]
        
        has_reply = False
        for offset in range(4):
            check_idx = first_seen + offset
            if check_idx < len(trajectory):
                if trajectory[check_idx].get("action_type") == "draft_reply" and trajectory[check_idx].get("message_id", check_idx) == um_id:
                    has_reply = True
                    break
        if has_reply:
            urgent_replied_in_time += 1
            
    priority_ordering_score = urgent_replied_in_time / max(1, total_urgent_messages)
    
    avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 1.0
    delegation_quality = 1.0 if avg_llm >= 0.70 else (avg_llm / 0.70)
    
    fp_rate = fp_spam / max(1, total_spam_flags) if total_spam_flags > 0 else 0.0
    classification_accuracy = 1.0 if fp_rate < 0.05 else max(0.0, 1.0 - fp_rate)
    
    clearance_rate = len(processed_msg_ids) / max(1, len(unique_msg_ids))
    if clearance_rate >= 0.90 and len(trajectory) <= 60:
        completion_rate = 1.0
    else:
        completion_rate = clearance_rate * (1.0 if len(trajectory) <= 60 else 0.5)
        
    return compute_final_score(
        classification_accuracy, 
        priority_ordering_score, 
        delegation_quality, 
        completion_rate, 
        reply_policy_violations=policy_violations
    )
