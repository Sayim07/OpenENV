---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags: [openenv, email, triage, rl-environment, agent-benchmark, business-automation]
datasets: []
license: mit
---

# Email Triage RL Environment

## 1. Overview
The **Email Triage RL Environment** is a procedural reinforcement learning benchmark designed to evaluate AI agents on complex, multi-level business process automation. Unlike simple classification tasks, this environment requires agents to manage an inbox dynamically, prioritizing high-value executive communications while filtering spam, delegating technical requests, and drafting policy-compliant replies.

### Key Stats:
- **3 Difficulty Levels**: Easy (Basic Triage), Medium (Priority Ranking), Hard (Drafting & Policy).
- **8 Discrete Actions**: Including delegation and asynchronous drafting.
- **12 Observation Fields**: Rich metadata for context-aware decision making.
- **Max Step Budget**: 30 (Easy), 45 (Medium), 60 (Hard).
- **Procedural Generation**: Billions of unique inbox variations via deterministic seeding.

---

## 2. Monorepo Installation
Requires **Python 3.10+** and **Node.js 18+**.

```bash
git clone https://huggingface.co/spaces/sayim/email-triage-env
cd email-triage-env
# Install Python package
pip install .
# Install Dashboard dependencies
cd dashboard && npm install && cd ..
```

---

## 3. Quick Start Dashboard & Website
Run the entire project (Monitoring Dashboard + RL Environment) with unified commands:

```bash
# Start the Dashboard (Port 3000)
npm run dev

# Start the RL Backend (Port 7860)
npm run backend
```
*Frontend: http://localhost:3000 | Backend: http://localhost:7860/health*

---

## 4. CLI Benchmark & Testing
```python
# Run RL Baseline Inference (Zero-Shot)
npm run baseline -- --task easy --episodes 5

# Run Full Test Suite (90%+ Coverage)
npm run test:env
```

---

## 5. Observation Space
The `Observation` object contains 12 structured fields:

| Field | Type | Description |
|---|---|---|
| `message_id` | `str` | Unique UUID for the message. |
| `subject` | `str` | Email subject (max 120 chars). |
| `sender_email` | `EmailStr` | Validated sender address. |
| `sender_tier` | `Enum` | executive, manager, peer, external, spam. |
| `body_snippet` | `str` | Snippet of email content (max 500 chars). |
| `thread_depth` | `int` | Integer (>=0) indicating conversation depth. |
| `has_attachment` | `bool` | Presence of attachments. |
| `received_at` | `datetime` | ISO timestamp of receipt. |
| `urgency_signal`| `float` | Heuristic signal [0.0, 1.0]. |
| `inbox_remaining`| `int` | Count of emails left in current session. |
| `step_budget_remaining` | `int` | Steps left before forced budget termination. |
| `context` | `dict` | Metadata/policy rules (Task-specific). |

---

## 5. Action Space
Agents must select one of the following 8 actions:

| Action | Payload Required | Description |
|---|---|---|
| `ARCHIVE` | None | Remove from inbox. |
| `LABEL_URGENT` | None | Mark as high priority. |
| `LABEL_DELEGATE`| `delegate_to` | Assign to a specific role/department. |
| `DRAFT_REPLY` | `reply_body` | Draft a response (Asynchronous in Hard mode). |
| `ESCALATE` | None | Flag for executive review. |
| `FLAG_SPAM` | None | Mark address as malicious/spam. |
| `SNOOZE` | None | Move message to end of the queue. |
| `NO_OP` | None | Skip current message without action. |

---

## 6. Reward Function
Rewards are additive per step and return a float in `[-1.0, 1.0]`:

| Component | Reward | Condition |
|---|---|---|
| **Correct Class** | `+0.30` | Action matches ground-truth label. |
| **Urgency Match** | `+0.20` | `LABEL_URGENT` on signal >= 0.8 email. |
| **Valid Delegation**| `+0.15` | `LABEL_DELEGATE` to correct internal role. |
| **Spam TP** | `+0.25` | `FLAG_SPAM` on authenticated spam sender. |
| **Reply Quality** | `+0.0-0.4` | LLM Grader Score * 0.40 (Hard Task). |
| **Incorrect Class** | `-0.20` | Mismatch vs ground-truth label. |
| **Spam FP** | `-0.35` | `FLAG_SPAM` on legitimate contact. |
| **Unjustified Esc** | `-0.25` | `ESCALATE` on non-critical/peer email. |
| **Loop Penalty** | `-0.15` | Per step from 4th consecutive `NO_OP`. |
| **Budget Penalty** | `-0.10` | Per step after soft budget limit. |
| **Policy Breach** | `-1.00` | Confidential data leak (Episode Terminates). |

---

## 7. Task Definitions
### Easy: Basic Triage
- **Success**: Archive newsletters, flag spam, and label executive mail as urgent.
- **Goal**: Clear 20 messages in 30 steps with high accuracy.

### Medium: Priority Ranking & Delegation
- **Success**: Identify 3 spoofed executives, delegate 5+ messages correctly, and maintain top-3 urgency ordering.
- **Goal**: Optimize role routing and handle ambiguous threads within 45 steps.

### Hard: Drafting & Policy Compliance
- **Success**: Draft replies to all urgent mail within 3 steps of receipt, maintaining 0.70+ quality score and 0% policy violations.
- **Goal**: Professional clearance of a 50-email inbox under strict allow/deny rules within 60 steps.

---

## 8. Grader Formula
The final episode score is computed as:
$$Score = (0.40 \times \text{Class Acc}) + (0.25 \times \text{Priority Score}) + (0.20 \times \text{Delegation Qual}) + (0.15 \times \text{Completion Rate})$$
*Hard Task additionally subtracts:* $0.10 \times \text{Policy Violations}$.

---

## 9. Running the Baseline
A zero-shot LLM inference agent is provided in `baseline.py`.

```bash
# Run 10 episodes on Easy task using gpt-4o-mini
export OPENAI_API_KEY="..."
python baseline.py --task easy --episodes 10 --model gpt-4o-mini
```

---

## 10. Baseline Scores
Results for **gpt-4o-mini** (Seed 42, N=10):
| Task | Mean Score | Std Dev |
|---|---|---|
| Easy | 0.62 | ± 0.05 |
| Medium | 0.41 | ± 0.08 |
| Hard | 0.28 | ± 0.11 |

---

## 11. Running Tests
The environment includes a 90%+ coverage test suite.

```bash
# Run all tests
pytest tests/

# Check coverage
pytest tests/ --cov=email_triage_env --cov-report=term-missing
```

---

## 12. Docker / HF Spaces Deployment
The environment is optimized for **Hugging Face Spaces**.

```bash
# Build locally
docker build -t email-triage-env .

# Run with Compose
docker-compose up
```
The health endpoint is available at `http://localhost:7860/health`.

---

## 13. Contributing
To extend the environment:
1. **New Tasks**: Add a YAML config in `tasks/`.
2. **Custom Rewards**: Implement logic in `reward.py` and register it in `env.py`.
3. **New Graders**: Add a function to `grader.py`.

Please ensure all new code maintains **90%+ test coverage**.
