import json
from datetime import datetime
import pytest
from email_triage_env import make
from email_triage_env.models import Action, ActionType, InvalidActionError

def test_make_env_reset() -> None:
    env = make("easy")
    obs = env.reset(seed=42)
    assert obs.inbox_remaining == 20
    assert obs.step_budget_remaining == 30

def test_step_logic() -> None:
    env = make("easy")
    env.reset(seed=42)
    
    action = Action(action_type=ActionType.ARCHIVE, payload={})
    obs, reward, done, info = env.step(action)
    assert not done
    assert obs.inbox_remaining == 19
    assert isinstance(reward, float)
    assert isinstance(info, dict)

def test_invalid_action() -> None:
    env = make("easy")
    env.reset()
    action = Action(action_type=ActionType.LABEL_DELEGATE, payload={"delegate_to": "HR"})
    del action.payload["delegate_to"]
    with pytest.raises(InvalidActionError):
        env.step(action)

def test_episode_terminates_inbox_empty() -> None:
    env = make("easy")
    env.reset(seed=1)
    for _ in range(19):
        env.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    obs, _, done, info = env.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    assert done
    assert "episode_score" in info

def test_episode_terminates_budget_exhausted() -> None:
    env = make("easy") 
    env.reset(seed=1)
    # 30 steps soft budget, hard budget is 45 (1.5x)
    for _ in range(44):
        _, _, done, _ = env.step(Action(action_type=ActionType.SNOOZE, payload={}))
        if done: break
    assert not done
    _, _, done, _ = env.step(Action(action_type=ActionType.SNOOZE, payload={}))
    assert done

def test_async_reward_logic_hard() -> None:
    env = make("hard")
    env.reset(seed=42)
    
    action = Action(action_type=ActionType.DRAFT_REPLY, payload={"reply_body": "test"})
    _, reward_1, done, _ = env.step(action)
    
    action2 = Action(action_type=ActionType.ARCHIVE, payload={})
    _, reward_2, done, _ = env.step(action2)
    assert reward_2 > 0.0 

def test_env_already_done() -> None:
    env = make("easy")
    env.reset(seed=1)
    for _ in range(20):
        env.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    # Now it's done
    _, _, done, _ = env.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    assert done
    # Step again
    _, r, d, info = env.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    assert d is True
    assert "error" in info

def test_env_snooze_cycle() -> None:
    env = make("easy")
    env.reset(seed=42)
    env.step(Action(action_type=ActionType.SNOOZE, payload={}))
    assert env._inbox[-1]["message_id"] is not None

def test_render() -> None:
    env = make("easy")
    env.reset(seed=42)
    result_text = env.render("text")
    assert "Inbox (20 messages remaining):" in result_text
    
    result_json = env.render("json")
    parsed = json.loads(result_json)
    assert len(parsed) == 20

def test_state_checkpoint() -> None:
    env = make("medium")
    env.reset(seed=42)
    state = env.state()
    assert state["task_level"] == "medium"
    assert "trajectory" in state

def test_determinism() -> None:
    env1 = make("easy")
    env1.reset(seed=42)
    _, r1, d1, _ = env1.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    
    env2 = make("easy")
    env2.reset(seed=42)
    _, r2, d2, _ = env2.step(Action(action_type=ActionType.ARCHIVE, payload={}))
    
    assert r1 == r2
    assert d1 == d2

def test_openenv_validate_compatibility() -> None:
    env = make("easy")
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "render")
    assert hasattr(env, "state")
