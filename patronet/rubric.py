"""Reward computation and verifiers for the Patronet environment.

Four dense reward functions, each a single boolean check.
One sparse reward function computing rescue outcome.
Three verifiers scoring the full trajectory on [0.0, 1.0].
"""


# ---------------------------------------------------------------------------
# Dense rewards, called by env.step after each action
# ---------------------------------------------------------------------------


def triage_reward(tag, crisis_type, asked, question_bank):
  """Returns +8 for relevant new tag, -5 otherwise."""
  valid_tags = question_bank.get(crisis_type, {})
  if tag in valid_tags and tag not in asked:
    return 8
  return -5


def routing_reward(responder_type, crisis_type, ontology):
  """Returns +20 for valid responder, -15 otherwise."""
  valid = ontology.get(crisis_type, {}).get("valid_responders", [])
  if responder_type in valid:
    return 20
  return -15


def idle_reward(victim_state):
  """Returns -15 when wait is called while victim is not stable or rescued."""
  if victim_state in ("deteriorating", "critical", "unresponsive"):
    return -15
  return 0


def deterioration_reward():
  """Returns -15 whenever victim state worsens by one level."""
  return -15


# ---------------------------------------------------------------------------
# Sparse rewards, called once at episode end
# ---------------------------------------------------------------------------


def compute_sparse_rewards(victim, responders):
  """Compute episode-end rewards based on final victim and responder state.
  Returns a dict of signal name to reward value.
  Precedence: rescued > partial > failure, mutually exclusive."""
  rewards = {}

  if victim["state"] == "rescued":
    rewards["rescue_success"] = 50
    return rewards

  # check if any responder is still en route
  en_route = any(r["status"] == "en_route" for r in responders)
  if en_route:
    rewards["rescue_partial"] = 20
    return rewards

  if victim["state"] == "unresponsive":
    rewards["rescue_failure"] = -50
    return rewards

  return rewards


# ---------------------------------------------------------------------------
# Verifiers, called at episode end for curriculum decisions
# ---------------------------------------------------------------------------


def compute_verifier_scores(trajectory, victim, responders, ontology, question_bank):
  """Returns dict with rescue, triage, routing scores in [0.0, 1.0]."""
  return {
    "rescue": _rescue_verifier(victim, responders, ontology),
    "triage": _triage_verifier(trajectory, victim, question_bank),
    "routing": _routing_verifier(trajectory, ontology),
  }


def _rescue_verifier(victim, responders, ontology):
  """1.0 if rescued within time window, 0.5 if partial, 0.0 if failure."""
  if victim["state"] == "rescued":
    return 1.0
  en_route = any(r["status"] == "en_route" for r in responders)
  if en_route:
    return 0.5
  return 0.0


def _triage_verifier(trajectory, victim, question_bank):
  """Score = relevant_asked / total_required, minus 0.1 per redundant, clamped to [0, 1]."""
  crisis = victim["crisis_type"]
  required_tags = set(question_bank.get(crisis, {}).keys())
  total_required = len(required_tags)
  if total_required == 0:
    return 1.0

  asked_tags = [
    entry[1]["question_tag"]
    for entry in trajectory
    if entry[1].get("tool") == "triage_assess"
  ]

  # relevant: tag in required set and first time asked
  seen = set()
  relevant_count = 0
  redundant_count = 0
  for tag in asked_tags:
    if tag in required_tags and tag not in seen:
      relevant_count += 1
      seen.add(tag)
    else:
      redundant_count += 1

  score = relevant_count / total_required - 0.1 * redundant_count
  return max(0.0, min(1.0, score))


def _routing_verifier(trajectory, ontology):
  """Score = correct_dispatches / total_dispatches, or 1.0 if none."""
  correct = 0
  total = 0
  for _, action, _, _ in trajectory:
    if action.get("tool") != "route_responder":
      continue
    total += 1
    crisis = "medical_emergency"
    valid = ontology.get(crisis, {}).get("valid_responders", [])
    if action["responder_type"] in valid:
      correct += 1

  if total == 0:
    return 1.0
  return correct / total
