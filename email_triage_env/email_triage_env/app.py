from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import email_triage_env

app = FastAPI(title="Email Triage RL Environment Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for real-time dashboard data
TELEMETRY: dict = {
    "total_emails": 0,
    "processed": 0,
    "success_rate": 0.0,
    "active_agent": "None",
    "logs": [],
    "recent_trajectories": [],
}


def _time_ago(iso_str: str) -> str:
    """Convert an ISO timestamp to a human-readable 'X ago' string."""
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


@app.get("/")
def read_root():
    return {
        "project": "Email Triage RL Environment",
        "status": "Running",
        "telemetry_active": True,
    }


@app.get("/stats")
def get_stats():
    """Return current telemetry with freshly computed time_ago for trajectories."""
    result = dict(TELEMETRY)
    # Recompute 'time' field for trajectories dynamically
    trajectories = []
    for t in TELEMETRY["recent_trajectories"]:
        entry = dict(t)
        entry["time"] = _time_ago(t.get("completed_at", ""))
        trajectories.append(entry)
    result["recent_trajectories"] = trajectories
    return result


@app.post("/report")
async def report_telemetry(data: dict):
    """
    Accepts telemetry updates from the environment or agent.
    Payload keys:
      - total_emails  (int)
      - processed     (int)
      - success_rate  (float)
      - active_agent  (str)
      - log           (str)  — single log line to append
    """
    global TELEMETRY
    for key in ["total_emails", "processed", "success_rate", "active_agent"]:
        if key in data:
            TELEMETRY[key] = data[key]

    # Append log line, keep last 100
    if "log" in data:
        TELEMETRY["logs"].append(data["log"])
        TELEMETRY["logs"] = TELEMETRY["logs"][-100:]

    return {"status": "success"}


@app.post("/trajectory")
async def add_trajectory(data: dict):
    """
    Called by the environment at episode end to record a completed trajectory.
    Expected payload:
      {
        "id":    "abc-123",
        "task":  "easy",
        "score": "0.84",
        "actions": ["ARCHIVE", "LABEL_URGENT", ...]   # optional
      }
    """
    global TELEMETRY
    entry = {
        "id": data.get("id", "???"),
        "task": data.get("task", "unknown"),
        "score": str(round(float(data.get("score", 0.0)), 2)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "actions": data.get("actions", []),
    }
    # most-recent first, keep last 10
    TELEMETRY["recent_trajectories"].insert(0, entry)
    TELEMETRY["recent_trajectories"] = TELEMETRY["recent_trajectories"][:10]
    return {"status": "success", "trajectory": entry}


@app.post("/reset")
async def reset_telemetry():
    """Clear all telemetry (useful between runs)."""
    global TELEMETRY
    TELEMETRY = {
        "total_emails": 0,
        "processed": 0,
        "success_rate": 0.0,
        "active_agent": "None",
        "logs": [],
        "recent_trajectories": [],
    }
    return {"status": "reset"}


@app.get("/health")
def health_check():
    """Returns HTTP 200 when env initializes cleanly."""
    try:
        env = email_triage_env.make()
        env.reset()
        return {"status": "ok", "message": "Environment initialized successfully"}
    except Exception as e:
        return Response(
            content=f"Error initializing environment: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
