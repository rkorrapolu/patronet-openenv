"""Patronet Environment adapting PatronetEnv to the OpenEnv interface."""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from patronet.models import PatronetAction, PatronetObservation
from patronet.env import PatronetEnv


class PatronetEnvironment(Environment):
  """OpenEnv-compatible wrapper around PatronetEnv."""

  SUPPORTS_CONCURRENT_SESSIONS: bool = True

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._env = PatronetEnv()
    self._state = State(episode_id=str(uuid4()), step_count=0)

  def reset(self, seed=None, episode_id=None, **kwargs) -> PatronetObservation:
    self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
    obs = self._env.reset()
    return self._to_observation(obs, reward=0.0, done=False)

  def step(self, action: PatronetAction, timeout_s=None, **kwargs) -> PatronetObservation:
    action_dict = {"tool": action.tool}
    if action.question_tag is not None:
      action_dict["question_tag"] = action.question_tag
    if action.victim_id is not None:
      action_dict["victim_id"] = action.victim_id
    if action.responder_type is not None:
      action_dict["responder_type"] = action.responder_type
    if action.priority is not None:
      action_dict["priority"] = action.priority
    if action.reason is not None:
      action_dict["reason"] = action.reason

    obs, reward, done, info = self._env.step(action_dict)
    self._state.step_count = obs["step_count"]

    extra_metadata = {}
    if done:
      extra_metadata["sparse_rewards"] = info.get("sparse_rewards", {})
      extra_metadata["verifier_scores"] = self._env.get_verifier_scores()

    return self._to_observation(obs, reward=reward, done=done, extra_metadata=extra_metadata)

  @property
  def state(self) -> State:
    return self._state

  def _to_observation(self, obs, reward, done, extra_metadata=None) -> PatronetObservation:
    return PatronetObservation(
      victims=obs["victims"],
      active_responders=obs["active_responders"],
      available_tools=obs["available_tools"],
      step_count=obs["step_count"],
      step_budget=obs["step_budget"],
      time_elapsed_seconds=obs["time_elapsed_seconds"],
      done=done,
      reward=reward,
      sparse_rewards=extra_metadata.get("sparse_rewards", {}) if extra_metadata else {},
      verifier_scores=extra_metadata.get("verifier_scores", {}) if extra_metadata else {},
    )
