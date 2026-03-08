"""Patronet OpenEnv environment server.

The FastAPI app and PatronetEnv class live here.
PatronetEnv manages episode state, the step loop, and delegates
reward computation to rubric.py.

Data structures:
  _victim: dict with state, crisis_type, language, questions_asked,
    time_since_intervention. Single victim for Level 1.
  _responders: list of dicts with type, eta_minutes, status.
  _trajectory: list of (state_before, action, reward, state_after) tuples,
    built up during the episode for verifier scoring at the end.
"""

import json
from pathlib import Path

from patronet import rubric

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ONTOLOGY = json.loads((DATA_DIR / "ontology.json").read_text(encoding="utf-8"))
QUESTION_BANK = json.loads((DATA_DIR / "triage.json").read_text(encoding="utf-8"))

# tool durations in seconds, from README time model
TOOL_DURATION = {
  "triage_assess": 15,
  "route_responder": 3,
  "wait": 15,
}

# dense_urban base ETA in minutes
BASE_ETA = 4.0

# medium pressure deterioration thresholds in seconds
DETERIORATION_THRESHOLDS = {
  "stable": 90,
  "deteriorating": 150,
  "critical": 300,
}

# ordered worsening for comparison
STATE_ORDER = ["stable", "deteriorating", "critical", "unresponsive", "rescued"]

STEP_BUDGET = 20


class PatronetEnv:
  """Emergency response environment for a single episode.

  Manages victim state, responder arrival, deterioration timing,
  and the step loop. Delegates reward computation to rubric functions.
  """

  def __init__(self, seed=42):
    self._seed = seed
    self._victim = None
    self._responders = []
    self._step_count = 0
    self._time_elapsed = 0.0
    self._done = False
    self._trajectory = []
    self._sparse_rewards = {}

  def reset(self):
    self._victim = {
      "state": "stable",
      "crisis_type": "medical_emergency",
      "language": "en",
      "questions_asked": [],
      "time_since_intervention": 0.0,
    }
    self._responders = []
    self._step_count = 0
    self._time_elapsed = 0.0
    self._done = False
    self._trajectory = []
    self._sparse_rewards = {}
    return self._observe()

  def step(self, action):
    tool = action["tool"]
    duration = TOOL_DURATION[tool]
    state_before = self._snapshot_victim()

    # D1: check responder arrival before executing action
    self._check_arrivals()
    if self._done:
      self._finalize()
      return self._observe(), 0, True, {"sparse_rewards": self._sparse_rewards}

    # execute the tool
    tool_reward = self._execute(action)

    # advance time
    self._time_elapsed += duration
    self._step_count += 1

    # subtract ETA on all en_route responders
    for r in self._responders:
      if r["status"] == "en_route":
        r["eta_minutes"] -= duration / 60.0

    # D1: deterioration check after action
    deterioration_reward = self._tick_deterioration(duration, action)

    reward = tool_reward + deterioration_reward
    state_after = self._snapshot_victim()
    self._trajectory.append((state_before, action, reward, state_after))

    # D1: budget check after step completes
    if self._step_count >= STEP_BUDGET:
      self._done = True
      self._finalize()

    info = {}
    if self._done:
      info["sparse_rewards"] = self._sparse_rewards

    return self._observe(), reward, self._done, info

  def get_verifier_scores(self):
    return rubric.compute_verifier_scores(
      self._trajectory,
      self._victim,
      self._responders,
      ONTOLOGY,
      QUESTION_BANK,
    )

  # -- tool handlers --------------------------------------------------------

  def _execute(self, action):
    tool = action["tool"]
    if tool == "triage_assess":
      return self._handle_triage(action)
    if tool == "route_responder":
      return self._handle_route(action)
    if tool == "wait":
      return self._handle_wait(action)
    return 0

  def _handle_triage(self, action):
    tag = action["question_tag"]
    crisis = self._victim["crisis_type"]
    asked = self._victim["questions_asked"]

    reward = rubric.triage_reward(tag, crisis, asked, QUESTION_BANK)
    self._victim["questions_asked"].append(tag)

    # any triage resets deterioration timer, per D3
    self._victim["time_since_intervention"] = 0.0
    return reward

  def _handle_route(self, action):
    responder_type = action["responder_type"]
    crisis = self._victim["crisis_type"]

    reward = rubric.routing_reward(responder_type, crisis, ONTOLOGY)

    self._responders.append({
      "type": responder_type,
      "eta_minutes": BASE_ETA,
      "status": "en_route",
    })

    # route_responder resets deterioration timer
    self._victim["time_since_intervention"] = 0.0
    return reward

  def _handle_wait(self, action):
    return rubric.idle_reward(self._victim["state"])

  # -- state transitions ----------------------------------------------------

  def _tick_deterioration(self, duration, action):
    """Advance deterioration timer unless action was a positive intervention.
    Triage and route already reset the timer in their handlers."""
    tool = action["tool"]

    # wait does not reset the timer
    if tool == "wait":
      self._victim["time_since_intervention"] += duration

    state_before = self._victim["state"]

    # unresponsive and rescued are terminal
    if state_before in ("unresponsive", "rescued"):
      return 0

    threshold = DETERIORATION_THRESHOLDS.get(state_before)
    if threshold is None:
      return 0

    if self._victim["time_since_intervention"] >= threshold:
      next_state = STATE_ORDER[STATE_ORDER.index(state_before) + 1]
      self._victim["state"] = next_state
      # reset timer for the next threshold
      self._victim["time_since_intervention"] = 0.0

      # unresponsive ends the episode
      if next_state == "unresponsive":
        self._done = True

      return rubric.deterioration_reward()

    return 0

  def _check_arrivals(self):
    """D1: arrival check at the start of each step."""
    for r in self._responders:
      if r["status"] == "en_route" and r["eta_minutes"] <= 0:
        r["status"] = "arrived"
        # rescue if victim is not already unresponsive
        if self._victim["state"] != "unresponsive":
          self._victim["state"] = "rescued"
          self._done = True

  def _finalize(self):
    """Compute sparse rewards at episode end."""
    self._sparse_rewards = rubric.compute_sparse_rewards(
      self._victim, self._responders
    )

  # -- observation ----------------------------------------------------------

  def _observe(self):
    victim_obs = {
      "state": self._victim["state"],
      "language": self._victim["language"],
      "message": None,
      "location": None,
      "threat_colocated": False,
    }

    responder_obs = [
      {"type": r["type"], "eta_minutes": r["eta_minutes"], "status": r["status"]}
      for r in self._responders
    ]

    return {
      "victims": [victim_obs],
      "active_responders": responder_obs,
      "available_tools": ["triage_assess", "route_responder", "wait"],
      "step_count": self._step_count,
      "step_budget": STEP_BUDGET,
      "time_elapsed_seconds": self._time_elapsed,
      "done": self._done,
    }

  def _snapshot_victim(self):
    return {
      "state": self._victim["state"],
      "questions_asked": list(self._victim["questions_asked"]),
      "time_since_intervention": self._victim["time_since_intervention"],
    }
