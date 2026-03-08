"""Tests for the Patronet environment, Level 1 mechanics.

Organized in order of specificity: observation shape, individual tool rewards,
deterioration model, responder arrival, episode boundaries, verifier scores,
then full integration.

All tests use the pinned default scenario:
  crisis_type=medical_emergency, location=dense_urban, victim=adult/english,
  infrastructure=full, responder_pool=well_resourced, time_pressure=medium,
  step_budget=20, threat_colocated=false, schema_drift=none.
"""

from patronet.env import PatronetEnv


# ---------------------------------------------------------------------------
# Section 1: Observation shape
# ---------------------------------------------------------------------------


class TestReset:
  def test_reset_returns_valid_observation(self):
    env = PatronetEnv(seed=42)
    obs = env.reset()

    assert isinstance(obs, dict)
    assert "victims" in obs
    assert "step_count" in obs
    assert "step_budget" in obs
    assert "done" in obs
    assert "available_tools" in obs

    assert obs["step_count"] == 0
    assert obs["step_budget"] == 20
    assert obs["done"] is False

    # single victim, starts stable, speaks english
    assert len(obs["victims"]) == 1
    victim = obs["victims"][0]
    assert victim["state"] == "stable"
    assert victim["language"] == "en"

    # level 1 tools only
    assert set(obs["available_tools"]) == {"triage_assess", "route_responder", "wait"}


# ---------------------------------------------------------------------------
# Section 2: Dense rewards for individual tools
# ---------------------------------------------------------------------------


class TestTriageReward:
  """triage_reward returns +8 for relevant new tag, -5 otherwise."""

  def test_valid_tag_rewards_plus_8(self):
    env = PatronetEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step({
      "tool": "triage_assess",
      "question_tag": "symptoms",
      "victim_id": 0,
    })
    assert reward == 8

  def test_invalid_tag_rewards_minus_5(self):
    """Tag not in the medical_emergency question bank."""
    env = PatronetEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step({
      "tool": "triage_assess",
      "question_tag": "water_depth",
      "victim_id": 0,
    })
    assert reward == -5

  def test_repeated_tag_rewards_minus_5(self):
    """Same valid tag asked twice becomes redundant."""
    env = PatronetEnv(seed=42)
    env.reset()
    env.step({"tool": "triage_assess", "question_tag": "symptoms", "victim_id": 0})

    obs, reward, done, info = env.step({
      "tool": "triage_assess",
      "question_tag": "symptoms",
      "victim_id": 0,
    })
    assert reward == -5


class TestRoutingReward:
  """routing_reward returns +20 for valid responder, -15 otherwise."""

  def test_correct_type_rewards_plus_20(self):
    env = PatronetEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step({
      "tool": "route_responder",
      "responder_type": "medical",
      "priority": 2,
      "victim_id": 0,
    })
    assert reward == 20

  def test_wrong_type_rewards_minus_15(self):
    env = PatronetEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step({
      "tool": "route_responder",
      "responder_type": "fire",
      "priority": 2,
      "victim_id": 0,
    })
    assert reward == -15


class TestWaitReward:
  """idle_reward returns -15 when victim is not stable, 0 when stable."""

  def test_wait_stable_victim_no_penalty(self):
    env = PatronetEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step({
      "tool": "wait",
      "reason": "thinking",
    })
    assert reward == 0

  def test_wait_deteriorating_victim_idle_penalty(self):
    """After 90s of no intervention, victim deteriorates.
    Next wait should incur idle penalty."""
    env = PatronetEnv(seed=42)
    env.reset()

    # 6 waits at 15s each = 90s, triggers stable to deteriorating
    for _ in range(6):
      env.step({"tool": "wait", "reason": "stalling"})

    # victim is now deteriorating, next wait should get -15 idle penalty
    # plus the deterioration reward may have fired on step 6
    obs, reward, done, info = env.step({"tool": "wait", "reason": "still stalling"})
    assert reward == -15
    assert obs["victims"][0]["state"] == "deteriorating"


# ---------------------------------------------------------------------------
# Section 3: Deterioration model
# ---------------------------------------------------------------------------


class TestDeterioration:
  def test_deterioration_after_90s_with_reward(self):
    """Medium pressure: stable to deteriorating after 90s, firing -15 reward.
    6 waits at 15s each = 90s exactly."""
    env = PatronetEnv(seed=42)
    env.reset()

    rewards = []
    for i in range(5):
      obs, r, _, _ = env.step({"tool": "wait", "reason": "stalling"})
      rewards.append(r)
      assert obs["victims"][0]["state"] == "stable", f"should be stable at step {i + 1}"

    # step 6 crosses the 90s threshold
    obs, r, _, _ = env.step({"tool": "wait", "reason": "stalling"})
    rewards.append(r)
    assert obs["victims"][0]["state"] == "deteriorating"
    assert rewards[5] == -15

  def test_triage_resets_deterioration_timer(self):
    """Any triage_assess resets the timer, even redundant ones, per decision D3."""
    env = PatronetEnv(seed=42)
    env.reset()

    # 5 waits = 75s, then a triage resets the timer
    for _ in range(5):
      env.step({"tool": "wait", "reason": "stalling"})

    # redundant triage at 75s resets the timer
    env.step({"tool": "triage_assess", "question_tag": "symptoms", "victim_id": 0})

    # 5 more waits = 75s from reset, should still be stable
    for _ in range(5):
      obs, _, _, _ = env.step({"tool": "wait", "reason": "stalling"})

    assert obs["victims"][0]["state"] == "stable"


# ---------------------------------------------------------------------------
# Section 4: Responder arrival
# ---------------------------------------------------------------------------


class TestResponderArrival:
  def test_responder_arrival_rescues_victim(self):
    """Dispatch at step 1, wait until arrival. With dense_urban ETA of 4 min
    and route_responder costing 3s, ETA starts at 3.95 min.
    Arrival happens when ETA crosses zero, checked at the start of each step."""
    env = PatronetEnv(seed=42)
    env.reset()

    obs, _, _, _ = env.step({
      "tool": "route_responder",
      "responder_type": "medical",
      "priority": 2,
      "victim_id": 0,
    })

    # responder should appear in active_responders
    assert len(obs["active_responders"]) == 1
    assert obs["active_responders"][0]["status"] == "en_route"

    # wait until done, which means rescue or budget exhaustion
    step_count = 1
    while not obs["done"]:
      obs, _, _, _ = env.step({"tool": "wait", "reason": "awaiting"})
      step_count += 1

    assert obs["victims"][0]["state"] == "rescued"
    # ETA 3.95 min, each wait subtracts 0.25 min
    # 3.95 / 0.25 = 15.8, so 16 waits needed, arrival at start of step 17
    assert step_count <= 20


# ---------------------------------------------------------------------------
# Section 5: Episode boundaries
# ---------------------------------------------------------------------------


class TestEpisodeBoundaries:
  def test_episode_ends_at_step_budget(self):
    """20 waits should exhaust the step budget."""
    env = PatronetEnv(seed=42)
    env.reset()

    for i in range(20):
      obs, _, done, _ = env.step({"tool": "wait", "reason": "exhausting budget"})

    assert done is True
    assert obs["step_count"] == 20

  def test_rescue_partial_when_en_route_at_end(self):
    """Dispatch late so responder is still en route when budget exhausts.
    Dispatch at step 18, only 2 steps remaining, nowhere near 4 min ETA.
    Sparse reward should include RESCUE_PARTIAL +20."""
    env = PatronetEnv(seed=42)
    env.reset()

    # burn 17 steps
    for _ in range(17):
      env.step({"tool": "wait", "reason": "stalling"})

    # dispatch at step 18
    env.step({
      "tool": "route_responder",
      "responder_type": "medical",
      "priority": 2,
      "victim_id": 0,
    })

    # steps 19-20 to exhaust budget
    for _ in range(2):
      obs, _, done, info = env.step({"tool": "wait", "reason": "awaiting"})

    assert done is True
    # responder should still be en route
    assert obs["active_responders"][0]["status"] == "en_route"
    # sparse rewards should include RESCUE_PARTIAL
    assert info["sparse_rewards"]["rescue_partial"] == 20


# ---------------------------------------------------------------------------
# Section 6: Verifier scores
# ---------------------------------------------------------------------------


class TestVerifiers:
  def test_routing_verifier_zero_on_wrong_dispatch(self):
    """Single wrong dispatch should give routing verifier 0.0."""
    env = PatronetEnv(seed=42)
    env.reset()

    env.step({
      "tool": "route_responder",
      "responder_type": "fire",
      "priority": 2,
      "victim_id": 0,
    })

    for _ in range(19):
      env.step({"tool": "wait", "reason": "x"})

    scores = env.get_verifier_scores()
    assert scores["routing"] == 0.0


# ---------------------------------------------------------------------------
# Section 7: Full integration
# ---------------------------------------------------------------------------


class TestFullEpisode:
  def test_full_episode_with_rescue(self):
    """Integration test: 2 triage, dispatch, redundant triage to keep stable,
    responder arrives, episode ends with rescue.
    Per the arithmetic trace in IMPLEMENTATION_PLAN.md:
      dispatch at step 3, arrival at start of step 20, total reward +6."""
    env = PatronetEnv(seed=42)
    env.reset()

    total_reward = 0

    # step 1: relevant triage
    _, r, _, _ = env.step({"tool": "triage_assess", "question_tag": "symptoms", "victim_id": 0})
    assert r == 8
    total_reward += r

    # step 2: relevant triage
    _, r, _, _ = env.step({"tool": "triage_assess", "question_tag": "breathing", "victim_id": 0})
    assert r == 8
    total_reward += r

    # step 3: dispatch
    _, r, _, _ = env.step({
      "tool": "route_responder",
      "responder_type": "medical",
      "priority": 2,
      "victim_id": 0,
    })
    assert r == 20
    total_reward += r

    # steps 4-19: triage to keep victim stable via timer reset
    # rotating through all 4 tags means consciousness and onset fire as
    # relevant on first use at steps 6-7, giving +8 each.
    # remaining 14 steps are redundant at -5 each.
    tags = ["symptoms", "breathing", "consciousness", "onset"]
    step = 4
    while True:
      obs, r, done, info = env.step({
        "tool": "triage_assess",
        "question_tag": tags[(step - 4) % len(tags)],
        "victim_id": 0,
      })
      total_reward += r

      # arrival fires at start of a step, so done may become true here
      if done:
        break
      step += 1

    assert obs["victims"][0]["state"] == "rescued"

    # sparse reward
    sparse = info.get("sparse_rewards", {})
    rescue_reward = sparse.get("rescue_success", 0)
    total_reward += rescue_reward
    assert rescue_reward == 50

    # dense: 2*8 + 20 + 2*8 + 14*(-5) = 16+20+16-70 = -18
    # sparse: +50
    # total: 32
    assert total_reward == 32

    scores = env.get_verifier_scores()
    assert scores["rescue"] == 1.0
    assert scores["routing"] == 1.0
    # triage verifier: 4 relevant out of 4 = 1.0, minus 12 redundant * 0.1 = -1.2
    # clamped to 0.0
    assert scores["triage"] == 0.0
