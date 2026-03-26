from typing import Any
from email_triage_env.models import Action, ActionType, SenderTier

def compute_reward(action: Action, ground_truth: dict[str, Any], step_context: dict[str, Any]) -> float:
    """
    Computes purely evaluating the reward step by step.
    
    Args:
        action: The Action chosen by the agent.
        ground_truth: Ground truth metadata dictionary.
        step_context: The episode-level state providing limits and flags.
        
    Returns:
        float: Additive numerical reward.
    """
    reward = 0.0
    
    gt_label = ground_truth.get("ground_truth_label")
    gt_tier = ground_truth.get("sender_tier")
    gt_urgency = ground_truth.get("urgency_signal", 0.0)
    
    task_level = step_context.get("task_level", "easy")
    
    # 1. Correct / Incorrect Classification
    action_label = {
        ActionType.ARCHIVE: "archive",
        ActionType.LABEL_URGENT: "urgent",
        ActionType.LABEL_DELEGATE: "delegate",
        ActionType.DRAFT_REPLY: "reply",
        ActionType.FLAG_SPAM: "spam",
    }.get(action.action_type)
    
    if action_label is not None:
        if action_label == gt_label:
            reward += 0.30
        else:
            reward -= 0.20
            
    # 2. Urgency tier match
    if action.action_type == ActionType.LABEL_URGENT and gt_urgency >= 0.8:
        reward += 0.20
        
    # 3. Valid delegation
    if action.action_type == ActionType.LABEL_DELEGATE:
        # Check payload matches the ground-truth expected delegate role
        expected_role = ground_truth.get("expected_delegate")
        if expected_role and action.payload.get("delegate_to") == expected_role:
            reward += 0.15
            
    # 4. Spam true positive & False positive spam flag
    if action.action_type == ActionType.FLAG_SPAM:
        if gt_label == "spam":
            reward += 0.25
        else:
            reward -= 0.35
            
    # 5. Reply quality (Hard)
    if action.action_type == ActionType.DRAFT_REPLY and task_level == "hard":
        llm_score = step_context.get("llm_grader_score", 0.0)
        reward += (llm_score * 0.40)
        
    # 6. Unjustified escalation
    if action.action_type == ActionType.ESCALATE:
        # Penalize if not an executive and urgency is low
        if gt_tier != SenderTier.EXECUTIVE.value and gt_urgency < 0.8:
            reward -= 0.25
            
    # 7. Loop penalty
    if action.action_type == ActionType.NO_OP:
        consecutive_before_this = step_context.get("consecutive_no_ops", 0)
        current_consecutive = consecutive_before_this + 1
        if current_consecutive >= 4:
            reward -= 0.15
            
    # 8. Budget overage
    if step_context.get("steps_over_budget", 0) > 0:
        reward -= 0.10
        
    # 9. Policy violation
    if action.action_type == ActionType.DRAFT_REPLY and task_level == "hard":
        if step_context.get("contains_confidential", False):
            reward -= 1.00
            # Mutate to signal the environment wrapper per instructions
            step_context["done"] = True
            
    return reward
