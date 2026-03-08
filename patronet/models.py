"""Data models for the Patronet Emergency Environment."""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class PatronetAction(Action):
  """Action for the Patronet environment."""

  tool: str = Field(..., description="Tool to invoke: triage_assess, route_responder, or wait")
  question_tag: Optional[str] = Field(default=None, description="Triage question tag")
  victim_id: Optional[int] = Field(default=None, description="Target victim index")
  responder_type: Optional[str] = Field(default=None, description="Type of responder to dispatch")
  priority: Optional[int] = Field(default=None, description="Dispatch priority level")
  reason: Optional[str] = Field(default=None, description="Reason for waiting")


class PatronetObservation(Observation):
  """Observation from the Patronet environment."""

  victims: List[dict] = Field(default_factory=list, description="List of victim observations")
  active_responders: List[dict] = Field(default_factory=list, description="Active responder states")
  available_tools: List[str] = Field(default_factory=list, description="Available tool names")
  step_count: int = Field(default=0, description="Current step number")
  step_budget: int = Field(default=20, description="Maximum steps per episode")
  time_elapsed_seconds: float = Field(default=0.0, description="Elapsed simulation time")
  sparse_rewards: dict = Field(default_factory=dict, description="Sparse rewards at episode end")
  verifier_scores: dict = Field(default_factory=dict, description="Verifier scores at episode end")
