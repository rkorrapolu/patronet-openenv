"""Patronet Emergency Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from patronet.models import PatronetAction, PatronetObservation


class PatronetEmergencyEnv(EnvClient[PatronetAction, PatronetObservation, State]):
  """Client for the Patronet Emergency Environment.

  Example:
      >>> with PatronetEmergencyEnv(base_url="http://localhost:8000") as client:
      ...     result = client.reset()
      ...     result = client.step(PatronetAction(tool="triage_assess", question_tag="symptoms", victim_id=0))
  """

  def _step_payload(self, action: PatronetAction) -> Dict:
    payload = {"tool": action.tool}
    if action.question_tag is not None:
      payload["question_tag"] = action.question_tag
    if action.victim_id is not None:
      payload["victim_id"] = action.victim_id
    if action.responder_type is not None:
      payload["responder_type"] = action.responder_type
    if action.priority is not None:
      payload["priority"] = action.priority
    if action.reason is not None:
      payload["reason"] = action.reason
    return payload

  def _parse_result(self, payload: Dict) -> StepResult[PatronetObservation]:
    obs_data = payload.get("observation", {})
    observation = PatronetObservation(
      victims=obs_data.get("victims", []),
      active_responders=obs_data.get("active_responders", []),
      available_tools=obs_data.get("available_tools", []),
      step_count=obs_data.get("step_count", 0),
      step_budget=obs_data.get("step_budget", 20),
      time_elapsed_seconds=obs_data.get("time_elapsed_seconds", 0.0),
      done=payload.get("done", False),
      reward=payload.get("reward"),
      sparse_rewards=obs_data.get("sparse_rewards", {}),
      verifier_scores=obs_data.get("verifier_scores", {}),
    )
    return StepResult(
      observation=observation,
      reward=payload.get("reward"),
      done=payload.get("done", False),
    )

  def _parse_state(self, payload: Dict) -> State:
    return State(
      episode_id=payload.get("episode_id"),
      step_count=payload.get("step_count", 0),
    )
