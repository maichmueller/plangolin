import csv
import logging
import warnings
from typing import Any, Dict, List

import pymimir as mi
import torch
from tensordict import TensorDict

from experiments.rl.data_layout import InputData, OutputData
from experiments.rl.thundeRL.cli_config import ThundeRLCLI
from rgnet import HeteroGraphEncoder
from rgnet.rl import (
    ActorCritic,
    EmbeddingModule,
    EmbeddingTransform,
    NonTensorTransformedEnv,
)
from rgnet.rl.envs import SuccessorEnvironment
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter


def eval_problem(test_problem: mi.Problem, agent: ActorCritic, max_steps: int):
    base_env = SuccessorEnvironment(
        generators=[mi.GroundedSuccessorGenerator(test_problem)],
        problems=[test_problem],
        batch_size=torch.Size((1,)),
    )
    env = NonTensorTransformedEnv(
        env=base_env,
        transform=EmbeddingTransform(
            current_embedding_key=agent.keys.current_embedding,
            env=base_env,
            embedding_module=agent.embedding_module,
        ),
        cache_specs=True,
    )
    rollout = env.rollout(
        max_steps=max_steps,
        policy=agent.as_td_module(
            env.keys.state, env.keys.transitions, env.keys.action, add_probs=True
        ),
    )
    return rollout


def _resolve_checkpoint_path(out_data: OutputData):
    checkpoint_dir = out_data.out_dir / "rgnet"
    dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if len(dirs) != 1:
        warnings.warn("Found more than one checkpoint directory.")
    checkpoint_paths = list((dirs[0] / "checkpoints").glob("*.ckpt"))
    if len(checkpoint_paths) != 1:
        warnings.warn("Found more than one checkpoint.")
    return checkpoint_paths[0]


def analyze_rollouts(results: Dict):

    out_data: List[Dict[str, Any]] = []
    rollout: TensorDict
    problem: mi.Problem
    for problem, rollout in results.items():
        action_probs = rollout["log_probs"].detach().exp()
        out_data.append(
            {
                "problem": problem.name,
                "solved": rollout["terminated"].any().item(),
                "rollout_length": rollout.batch_size[-1],
                "average_certainty": round(action_probs.mean().item(), 4),
                "min_certainty": round(action_probs.min().item(), 4),
                "action_sequence": [
                    transition.action for transition in rollout["action"][0]
                ],
            }
        )
    return out_data


def test_lightning_agent():
    # TODO disable WandbLogger or make it re-init
    cli = ThundeRLCLI(run=False)
    lightning_adapter: LightningAdapter = cli.model
    in_data: InputData = cli.datamodule.data
    # TODO
    out_data = cli.config_init["data_layout.output_data"]
    checkpoint_path = _resolve_checkpoint_path(out_data=out_data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lightning_adapter.load_state_dict(checkpoint["state_dict"])

    agent = lightning_adapter.actor_critic
    embedding_module = EmbeddingModule(
        encoder=HeteroGraphEncoder(in_data.domain), gnn=lightning_adapter.gnn
    )
    # TODO
    agent._embedding_module = embedding_module
    max_steps = cli.config_init["test_max_steps"]
    if not in_data.test_problems:
        raise ValueError("No test instances provided")
    test_instances = in_data.test_problems

    test_results = {}

    for test_problem in test_instances:
        rollout = eval_problem(test_problem, agent, max_steps=max_steps)
        test_results[test_problem] = rollout
        logging.info("Completed " + str(test_problem.name))

    analyzed_data = analyze_rollouts(test_results)

    results_file = out_data.out_dir / "results.csv"
    with open(results_file, "w") as f:
        writer = csv.DictWriter(f, analyzed_data[0].keys())
        writer.writeheader()
        writer.writerows(analyzed_data)
    logging.info("Saved results to " + str(results_file))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    test_lightning_agent()
