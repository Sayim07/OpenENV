import json
from datetime import datetime
from typing import Any, Optional, Literal
import requests

from email_triage_env.models import Action, ActionType, InvalidActionError, Observation, SenderTier
from email_triage_env.generator import generate_inbox
from email_triage_env.reward import compute_reward
from email_triage_env.grader import grade_easy, grade_medium, grade_hard


class EmailTriageEnv:
    """
    OpenEnv-compliant core environment for executing Email Triage tasks.
    Evaluates LLM Agent actions iteratively and computes rewards based on defined policies.
    """
    def __init__(self, task: str = "easy") -> None:
        """
        Initializes the environment for a specific task level mapping to budget variations.
        """
        self.task_level: str = task
        if task == "easy":
            self.max_steps = 30
        elif task == "medium":
            self.max_steps = 45
        else:
            self.max_steps = 60
            
        self._inbox: list[dict[str, Any]] = []
        self._trajectory: list[dict[str, Any]] = []
        self._current_msg_idx: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._consecutive_no_ops: int = 0
        self._pending_async_eval: Optional[str] = None
        self._telemetry_url = "http://127.0.0.1:7860/report"
        self._trajectory_url = "http://127.0.0.1:7860/trajectory"
        self._total_reward: float = 0.0
        
    def _emit_telemetry(self, log_msg: str | None = None, active_agent: str | None = None):
        """Send current state to dashboard."""
        try:
            # Calculate success_rate based on current _total_reward and _step_count
            reward_sum = float(self._total_reward)
            step_divisor = float(max(1, self._step_count))
            success_rate = round((reward_sum / step_divisor) * 100, 1)
            
            payload = {
                "total_emails": len(self._inbox),
                "processed": self._step_count,
                "success_rate": success_rate,
                "active_agent": active_agent if active_agent else "Autonomous Agent"
            }
            if log_msg:
                payload["log"] = log_msg
            requests.post(self._telemetry_url, json=payload, timeout=0.1)
        except Exception:
            # Suppress exceptions to avoid breaking the environment if telemetry fails
            pass

    def _emit_trajectory(self, score: float, actions: list[str]) -> None:
        """POST completed episode data to /trajectory for the Recent Trajectories panel."""
        import uuid
        try:
            payload = {
                "id": uuid.uuid4().hex[:7],
                "task": self.task_level,
                "score": round(score, 2),
                "actions": actions,
            }
            requests.post(self._trajectory_url, json=payload, timeout=0.5)
        except Exception:
            pass

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Resets the environment and inbox entirely deterministically from a designated seed.
        
        Args:
            seed: Arbitrary random seed ensuring pure reproducibility.
            
        Returns:
            The first Observation representing the earliest email payload.
        """
        self._inbox = generate_inbox(size=20, seed=seed, task_level=self.task_level)
        self._trajectory = []
        self._current_msg_idx = 0
        self._step_count = 0
        self._done = False
        self._consecutive_no_ops = 0
        self._pending_async_eval = None
        self._total_reward = 0.0 # Reset total reward on episode reset
        self._emit_telemetry(f"Reset: {len(self._inbox)} messages generated.")
        
        return self._get_observation()
        
    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Executes an agent's discrete action on the environment resulting in advancement.
        
        Args:
            action: Encapsulated intent dictating consequence processing constraints.
            
        Returns:
            tuple combining [next_observation, cumulative_reward, done_status, info_dict].
            
        Raises:
            InvalidActionError: Should semantic payloads missing required roles be identified.
        """
        if self._done:
            return self._get_dummy_observation(), 0.0, True, {"error": "Episode sequence previously exhausted."}

        # Malformed action semantics enforcement per specification
        if action.action_type == ActionType.LABEL_DELEGATE and "delegate_to" not in action.payload:
            raise InvalidActionError("Malformed payload condition: 'delegate_to' definitively omitted.")
        if action.action_type == ActionType.DRAFT_REPLY and "reply_body" not in action.payload:
            raise InvalidActionError("Malformed payload condition: 'reply_body' definitively omitted.")

        current_msg = self._inbox[self._current_msg_idx]
        
        # SNOOZE automatically buffers messages indefinitely to the terminus
        if action.action_type == ActionType.SNOOZE:
            self._inbox.append(current_msg)

        if action.action_type == ActionType.NO_OP:
            self._consecutive_no_ops += 1
        else:
            self._consecutive_no_ops = 0

        # Policy compliance boundary bounds
        contains_confidential = False
        reply_body = str(action.payload.get("reply_body", ""))
        if "secret" in reply_body.lower() or "confidential" in reply_body.lower():
            contains_confidential = True

        soft_budget = self.max_steps
        hard_budget = int(self.max_steps * 1.5)

        step_context: dict[str, Any] = {
            "task_level": self.task_level,
            "consecutive_no_ops": max(0, self._consecutive_no_ops - 1) if action.action_type == ActionType.NO_OP else 0,
            "steps_over_budget": max(0, (self._step_count + 1) - soft_budget),
            "contains_confidential": contains_confidential,
            "llm_grader_score": 0.0
        }
        
        # Asynchronous decoupled LLM processing component requirement abstraction
        async_reward = 0.0
        if self._pending_async_eval is not None:
            mock_llm_score = 0.85 
            async_reward = mock_llm_score * 0.40
            if self._trajectory:
                self._trajectory[-1]["llm_grader_score"] = mock_llm_score
            self._pending_async_eval = None

        if action.action_type == ActionType.DRAFT_REPLY and self.task_level == "hard":
            self._pending_async_eval = reply_body

        # Apply purely additive functional reward evaluations
        reward = compute_reward(action, current_msg, step_context)
        reward += async_reward

        self._step_count += 1
        
        t_entry = {
            "action_type": action.action_type.value,
            "payload": action.payload,
            "ground_truth_label": current_msg.get("ground_truth_label"),
            "sender_tier": current_msg.get("sender_tier"),
            "urgency_signal": current_msg.get("urgency_signal"),
            "message_id": current_msg.get("message_id"),
            "subject": current_msg.get("subject"),
            "expected_delegate": current_msg.get("expected_delegate", "HR"),
            "contains_confidential": contains_confidential,
            "llm_grader_score": 0.0
        }
        self._trajectory.append(t_entry)

        self._current_msg_idx += 1
        
        # Expiry thresholds determining episode cessation states
        done = False
        if self._current_msg_idx >= len(self._inbox):
            done = True
        if self._step_count >= hard_budget:
            done = True
        if step_context.get("done", False):
            done = True
            
        self._done = done
        self._total_reward += reward

        info: dict[str, Any] = {}
        if done:
            episode_score = self._compute_episode_score()
            info["episode_score"] = episode_score
            # Collect action summary for trajectory record
            action_types = [t["action_type"] for t in self._trajectory]
            self._emit_trajectory(episode_score, action_types)

        if done:
            next_obs = self._get_dummy_observation()
        else:
            next_obs = self._get_observation()

        self._emit_telemetry(f"Action: {action.action_type.value} | Reward: {reward:.2f}")

        return next_obs, reward, self._done, info
        
    def state(self) -> dict[str, Any]:
        """
        Compiles the entire application schema into an architecture capable 
        of immediate JSON encoding.
        """
        inbox_copy = []
        for m in self._inbox:
            m_copy = m.copy()
            if "received_at" in m_copy and isinstance(m_copy["received_at"], datetime):
                m_copy["received_at"] = m_copy["received_at"].isoformat()
            inbox_copy.append(m_copy)

        return {
            "task_level": self.task_level,
            "max_steps": self.max_steps,
            "current_msg_idx": self._current_msg_idx,
            "step_count": self._step_count,
            "done": self._done,
            "consecutive_no_ops": self._consecutive_no_ops,
            "pending_async_eval": self._pending_async_eval,
            "inbox": inbox_copy,
            "trajectory": [t.copy() for t in self._trajectory],
        }
        
    def render(self, mode: Literal["text", "json"] = "text") -> str:
        """
        Reflects current environment conditions visually or programmatically.
        
        Args:
            mode: Enum determining text terminal strings versus backend schema values.
        """
        remaining = self._inbox[self._current_msg_idx:]
        if mode == "json":
            def default_serializer(o: Any) -> Any:
                if isinstance(o, datetime):
                    return o.isoformat()
                raise TypeError(f"Type {type(o)} not serializable")
            return json.dumps(remaining, default=default_serializer)
        else:
            if not remaining:
                return "Inbox is empty."
            output = [f"Inbox ({len(remaining)} messages remaining):"]
            for idx, m in enumerate(remaining):
                subj = m.get('subject', '')
                email = m.get('sender_email', '')
                tier = m.get('sender_tier', '')
                urgency = float(m.get('urgency_signal', 0.0))
                output.append(f"[{idx+1}] {subj} from {email} (Tier: {tier}, Urgency: {urgency:.2f})")
            return "\n".join(output)

    def _get_observation(self) -> Observation:
        msg = self._inbox[self._current_msg_idx]
        return Observation(
            message_id=msg["message_id"],
            subject=msg["subject"],
            sender_email=msg["sender_email"],
            sender_tier=SenderTier(msg["sender_tier"]),
            body_snippet=msg["body_snippet"],
            thread_depth=msg["thread_depth"],
            has_attachment=msg["has_attachment"],
            received_at=msg["received_at"],
            urgency_signal=msg["urgency_signal"],
            inbox_remaining=len(self._inbox) - self._current_msg_idx,
            step_budget_remaining=max(0, self.max_steps - self._step_count),
            context={}
        )

    def _get_dummy_observation(self) -> Observation:
        return Observation(
            message_id="done",
            subject="done",
            sender_email="done@done.com",
            sender_tier=SenderTier.EXTERNAL,
            body_snippet="done",
            thread_depth=0,
            has_attachment=False,
            received_at=datetime.now(),
            urgency_signal=0.0,
            inbox_remaining=0,
            step_budget_remaining=max(0, self.max_steps - self._step_count),
            context={"done": True}
        )

    def _compute_episode_score(self) -> float:
        if self.task_level == "easy":
            return grade_easy(self._trajectory)
        elif self.task_level == "medium":
            return grade_medium(self._trajectory)
        else:
            return grade_hard(self._trajectory)

def make(task: str = "easy") -> EmailTriageEnv:
    """Instantiation factory logic."""
    return EmailTriageEnv(task)
