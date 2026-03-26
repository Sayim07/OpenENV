import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import email_triage_env

app = FastAPI(title="OpenEnv Email Triage Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for real-time dashboard telemetry
TELEMETRY: dict = {
    "total_emails": 0,
    "processed": 0,
    "success_rate": 0.0,
    "active_agent": "None",
    "logs": [],
    "recent_trajectories": [],
}

def _time_ago(iso_str: str) -> str:
    try:
        then = datetime.fromisoformat(iso_str)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - then
        s = int(delta.total_seconds())
        if s < 60:
            return f"{s}s ago"
        elif s < 3600:
            return f"{s // 60}m ago"
        else:
            return f"{s // 3600}h ago"
    except Exception:
        return "just now"

# Global environment instance
global_env = None

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: str

@app.get("/health")
def health_check():
    """Returns HTTP 200"""
    return {"status": "ok"}

@app.post("/reset")
def env_reset(req: ResetRequest):
    global global_env
    if global_env is None:
        global_env = email_triage_env.make()
    
    obs, info = global_env.reset(seed=req.seed, options={"task": req.task})
    
    # reset telemetry on fresh start
    global TELEMETRY
    TELEMETRY.update({
        "total_emails": info.get("total_emails", 0) if info else 0,
        "processed": 0,
        "success_rate": 0.0,
        "logs": ["# Environment Reset"]
    })
    
    return {"observation": obs, "info": info}

@app.post("/step")
def env_step(req: StepRequest):
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    
    obs, reward, done, truncated, info = global_env.step(req.action)
    
    # Update telemetry
    global TELEMETRY
    TELEMETRY["processed"] += 1
    TELEMETRY["logs"].append(f"Action: {req.action} | Reward: {reward}")
    TELEMETRY["logs"] = TELEMETRY["logs"][-100:]
    
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
def get_state():
    global global_env
    if global_env is None:
        return {"state": None}
    try:
        return {"state": global_env.state}
    except AttributeError:
        return {"state": "Environment is initialized."}

# Telemetry endpoints for frontend dashboard
@app.get("/stats")
def get_stats():
    result = dict(TELEMETRY)
    trajectories = []
    for t in TELEMETRY["recent_trajectories"]:
        entry = dict(t)
        entry["time"] = _time_ago(t.get("completed_at", ""))
        trajectories.append(entry)
    result["recent_trajectories"] = trajectories
    return result

@app.post("/report")
async def report_telemetry(data: dict):
    global TELEMETRY
    for key in ["total_emails", "processed", "success_rate", "active_agent"]:
        if key in data:
            TELEMETRY[key] = data[key]
    if "log" in data:
        TELEMETRY["logs"].append(data["log"])
        TELEMETRY["logs"] = TELEMETRY["logs"][-100:]
    return {"status": "success"}

@app.post("/trajectory")
async def add_trajectory(data: dict):
    global TELEMETRY
    entry = {
        "id": data.get("id", "???"),
        "task": data.get("task", "unknown"),
        "score": str(round(float(data.get("score", 0.0)), 2)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "actions": data.get("actions", []),
    }
    TELEMETRY["recent_trajectories"].insert(0, entry)
    TELEMETRY["recent_trajectories"] = TELEMETRY["recent_trajectories"][:10]
    return {"status": "success", "trajectory": entry}

# Serve the Next.js static build from dashboard/out
# --- Static File Serving (Must be LAST to avoid swallowing API routes) ---
if os.path.exists("dashboard/out"):
    app.mount("/", StaticFiles(directory="dashboard/out", html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return {"message": "Frontend build not available. Call /health or /reset."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

