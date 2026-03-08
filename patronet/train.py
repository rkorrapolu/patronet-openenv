"""Stage 7: GRPOTrainer training script for Patronet Emergency Environment.

Purple agent (LLM under training) runs here on Northflank H100.
Green agent (environment server) runs on HF Spaces.

Usage:
  python -m patronet.train \
    --env_url https://freddy-0-patronet-emergency.hf.space \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num_episodes 256

Architecture:
  - GRPOTrainer generates completions (action plans) from the policy model.
  - The reward function parses each completion as a sequence of JSON actions,
    executes them against the remote environment, and returns the total reward.
  - No rollout_func needed — reward_fn handles the environment interaction.
"""

import argparse
import json
import logging
import re

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from patronet.env import PatronetEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -- System prompt for the emergency response agent ----------------------------

SYSTEM_PROMPT = """\
You are an emergency response coordinator managing a medical emergency.

## Victim
A single victim with state=stable, crisis_type=medical_emergency.

## Available Tools (output one JSON action per line)
- {"tool": "triage_assess", "question_tag": "<tag>", "victim_id": 0}
  Valid tags: symptoms, consciousness, breathing, onset
- {"tool": "route_responder", "responder_type": "medical", "victim_id": 0}
- {"tool": "wait", "reason": "<reason>"}

## Strategy
1. Ask all 4 triage questions (symptoms, consciousness, breathing, onset)
2. Dispatch a medical responder
3. Wait for responder arrival

## Output
Output your complete action plan as one JSON action per line. No other text."""


def parse_actions(text: str) -> list[dict]:
  """Extract all JSON action objects from completion text."""
  actions = []
  for match in re.finditer(r"\{[^}]+\}", text):
    try:
      action = json.loads(match.group())
      if "tool" in action:
        actions.append(action)
    except json.JSONDecodeError:
      continue
  return actions


def score_action_plan(actions: list[dict]) -> float:
  """Run a sequence of actions against a local PatronetEnv and return total reward."""
  env = PatronetEnv()
  env.reset()

  total_reward = 0.0
  for action_dict in actions:
    tool = action_dict.get("tool")
    if tool not in ("triage_assess", "route_responder", "wait"):
      total_reward -= 5.0
      continue

    try:
      _, reward, done, info = env.step(action_dict)
      total_reward += reward
      if done:
        total_reward += sum(info.get("sparse_rewards", {}).values())
        break
    except (KeyError, ValueError):
      total_reward -= 5.0
      continue

  return total_reward


def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
  """Parse each completion as an action plan, run against env, return rewards."""
  rewards = []
  for completion in completions:
    actions = parse_actions(completion)
    if not actions:
      rewards.append(-20.0)
      continue
    score = score_action_plan(actions)
    rewards.append(score)
  return rewards


def build_prompt_dataset(num_episodes: int) -> Dataset:
  """Build dataset of prompts for training."""
  return Dataset.from_dict({"prompt": [SYSTEM_PROMPT] * num_episodes})


def main():
  parser = argparse.ArgumentParser(description="Train Patronet emergency response agent with GRPO")
  parser.add_argument("--env_url", type=str, default="https://freddy-0-patronet-emergency.hf.space")
  parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
  parser.add_argument("--num_episodes", type=int, default=256)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--num_generations", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=5e-7)
  parser.add_argument("--num_epochs", type=int, default=1)
  parser.add_argument("--output_dir", type=str, default="./patronet-grpo-output")
  args = parser.parse_args()

  logger.info("Building prompt dataset with %d episodes", args.num_episodes)
  dataset = build_prompt_dataset(args.num_episodes)

  logger.info("Configuring GRPOTrainer with model %s", args.model)
  config = GRPOConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_generations=args.num_generations,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_epochs,
    logging_steps=1,
    save_steps=50,
    max_completion_length=512,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
  )

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  trainer = GRPOTrainer(
    model=args.model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
  )

  logger.info("Starting GRPO training against local env (model on GPU)")
  trainer.train()

  logger.info("Saving model to %s", args.output_dir)
  trainer.save_model(args.output_dir)
  tokenizer.save_pretrained(args.output_dir)

  logger.info("Training complete!")


if __name__ == "__main__":
  main()
