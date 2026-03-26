import time
import random
import uuid
import requests
from datetime import datetime, timezone

TELEMETRY_URL = "http://127.0.0.1:7860/report"
TRAJECTORY_URL = "http://127.0.0.1:7860/trajectory"

ACTIONS = ["ARCHIVE", "LABEL_URGENT", "LABEL_DELEGATE", "DRAFT_REPLY", "ESCALATE", "FLAG_SPAM", "SNOOZE", "NO_OP"]

def run_mock_episode(task_level: str, model_name: str, episode_num: int):
    print(f"\n▶ Starting Episode {episode_num} ({task_level} difficulty)")
    
    # 1. Reset / initialize parameters
    total_emails = random.randint(15, 25)
    processed = 0
    score = 0.0
    action_history = []
    
    # 2. Simulate processing each step
    steps = random.randint(10, total_emails + 5)
    for step in range(steps):
        # Pick a random action
        action = random.choice(ACTIONS)
        action_history.append(action)
        
        # Earn some reward (or penalty)
        reward = random.uniform(-0.1, 0.5)
        if action == "NO_OP":
            reward = -0.2
        score += reward
        processed += 1
        
        # Compute live success rate percentage
        success_rate = round((score / processed) * 100, 1)
        
        # Log msg
        log_msg = f"Action: {action} | Reward: {reward:.2f}"
        
        # Push telemetry step
        payload = {
            "total_emails": total_emails,
            "processed": processed,
            "success_rate": success_rate,
            "active_agent": model_name,
            "log": log_msg
        }
        try:
            requests.post(TELEMETRY_URL, json=payload, timeout=0.5)
        except Exception:
            pass
            
        print(f"  {log_msg}")
        time.sleep(random.uniform(0.3, 0.8))  # Thinking delay

    # 3. Episode done - push trajectory
    print(f"✓ Episode complete! Final Score: {score:.2f}")
    try:
        traj_payload = {
            "id": uuid.uuid4().hex[:7],
            "task": task_level,
            "score": round(score, 2),
            "actions": action_history
        }
        requests.post(TRAJECTORY_URL, json=traj_payload, timeout=0.5)
        
        # Reset telemetry payload to clear logs gracefully for next run
        requests.post(TELEMETRY_URL, json={"log": f"Reset: {total_emails} messages generated."}, timeout=0.5)

    except Exception:
        pass


if __name__ == "__main__":
    print("🚀 Mock Baseline Agent Initialized")
    print("Ensure the FastAPI backend (app.py) is running on port 7860.")
    
    # Send an initial reset pulse
    try:
        requests.post("http://127.0.0.1:7860/reset", timeout=1.0)
    except:
        pass

    for i in range(1, 100):
        run_mock_episode("easy", "mock-agent-fast", i)
        time.sleep(2)
        
    print("\n🎉 Mock simulation finished!")
