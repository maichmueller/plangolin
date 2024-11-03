import csv
import dataclasses
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pymimir as mi
import torch
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import Logger, WandbLogger
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type

from experiments.rl.data_layout import InputData, OutputData
from experiments.rl.thundeRL.cli_config import TestSetup, ThundeRLCLI
from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl import (
    ActorCritic,
    EmbeddingModule,
    EmbeddingTransform,
    NonTensorTransformedEnv,
)
from rgnet.rl.envs import SuccessorEnvironment
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter


# TODO Unify with experiments.analyze_run.PlanResult
@dataclasses.dataclass
class ProbabilisticPlanResult:
    problem: mi.Problem
    cost: float = dataclasses.field(init=False)  # derived from action_sequence
    action_sequence: List[mi.Action]
    solved: bool
    average_probability: float
    min_probability: float
    solved_optimal: Optional[bool] = None

    def __post_init__(self):
        self.cost = sum(action.cost for action in self.action_sequence)

    # cant use dataclasses.asdict(...) because pymimir problems can't be pickled
    def serialize_as_dict(self):
        def transform(k, v):
            if isinstance(v, mi.Problem):
                return v.name
            elif k == "action_sequence":
                return str(v)
            return v

        return {
            f.name: transform(f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
        }


def rollout_on_problem(
    test_problem: mi.Problem, agent: ActorCritic, test_setup: TestSetup
):
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
    with set_exploration_type(test_setup.exploration_type):
        rollout = env.rollout(
            max_steps=test_setup.max_steps,
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


def _analyze_rollouts(
    results: Dict[mi.Problem, TensorDict]
) -> List[ProbabilisticPlanResult]:
    out_data: List[ProbabilisticPlanResult] = []
    rollout: TensorDict

    problem: mi.Problem
    for problem, rollout in results.items():
        # Assert we only have one batch entry and the time dimension is the last
        assert rollout.batch_size[0] == 1
        assert rollout.names[-1] == "time"
        action_probs = rollout["log_probs"].detach().exp()
        out_data.append(
            ProbabilisticPlanResult(
                problem=problem,
                solved=rollout[("next", "terminated")].any().item(),
                average_probability=round(action_probs.mean().item(), 4),
                min_probability=round(action_probs.min().item(), 4),
                action_sequence=[
                    transition.action for transition in rollout["action"][0]
                ],
            )
        )
    return out_data


def wandb_id_resolver(out_data: OutputData) -> str:
    """
    Try to find the wandb run id from the output directory.
    First look in the wandb directory, where we hope to find the following
    wandb
        run-<time_stamp>-<run_id>
        ...
    Otherwise, we look for the lightning checkpoint directory.
    rgnet
        run_id
            checkpoints
    """
    wandb_dir = out_data.out_dir / "wandb"
    if wandb_dir.is_dir():
        run_dir = next(wandb_dir.glob("run-*"), None)
        if run_dir is not None:
            run_id = run_dir.name.split("-")[-1]
            if len(run_id) == 8:
                return run_id
    # try the lighting logging directory
    if (lightning_dir := out_data.out_dir / "rgnet").is_dir():
        run_dir = next(lightning_dir.iterdir(), None)
        if run_dir is not None and len(run_dir.name) == 8:
            return run_dir.name


class TestThundeRLCLI(ThundeRLCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        # fit subcommand adds this value to the config
        parser.add_argument("--ckpt_path", type=Optional[Path], default=None)
        parser.link_arguments(
            "data_layout.output_data",
            "trainer.logger.init_args.id",
            compute_fn=wandb_id_resolver,
            apply_on="instantiate",
        )


def test_lightning_agent(
    lightning_adapter: LightningAdapter,
    logger: Logger,
    input_data: InputData,
    output_data: OutputData,
    test_setup: TestSetup,
):
    """
    Run the learned agent of every test problem specified in the input_data.
    The agent is run once on each problem until either max_steps are reached or a terminal state is encountered.
    There is currently no GPU support as the memory transfer is often more significant as
    the relatively small network forward passes.
    The action can either be sampled from the probability distribution using ExplorationType.RANDOM
    or the argmax is used (ExplorationType.MODE).
    There is no cycle detection implemented.
    Note that the terminated signal is only emitted after a transition from a goal state.
    The output is saved as csv under the out directory of OutputData as results_{epoch}_{step}.csv
    referencing the epoch and step form the loaded checkpoint.

    :param lightning_adapter: An agent instance. The weights for the agent will be loaded from a checkpoint.
    :param logger: If a WandbLogger is passed the results are uploaded as table.
    :param input_data: InputData which should specify at least one test_problem.
    :param output_data: OutputData pointing to the checkpoint containing the learned weights for the agent.
    :param test_setup: Extra parameter for testing the agent.
    """
    if not input_data.test_problems:
        raise ValueError("No test instances provided")

    checkpoint_path = _resolve_checkpoint_path(out_data=output_data)
    epoch, step = re.match(r"epoch=(\d+)-step=(\d+)", checkpoint_path.name).groups()
    logging.info(f"Using checkpoint with {epoch=}, {step=}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lightning_adapter.load_state_dict(checkpoint["state_dict"])

    agent = lightning_adapter.actor_critic
    embedding_module = EmbeddingModule(
        encoder=HeteroGraphEncoder(input_data.domain), gnn=lightning_adapter.gnn
    )
    # TODO fix embedding from agent, can't mix properties and torch.nn.Module
    agent._embedding_module = embedding_module
    test_instances = input_data.test_problems

    test_results = {}

    for test_problem in test_instances:
        rollout = rollout_on_problem(test_problem, agent, test_setup)
        test_results[test_problem] = rollout
        logging.info("Completed " + str(test_problem.name))

    analyzed_data: List[ProbabilisticPlanResult] = _analyze_rollouts(test_results)

    results_name = f"results_epoch={epoch}-step={step}"
    results_file = output_data.out_dir / (results_name + ".csv")
    plan_results_as_dict = [
        plan_result.serialize_as_dict() for plan_result in analyzed_data
    ]
    with open(results_file, "w") as f:
        writer = csv.DictWriter(
            f,
            plan_results_as_dict[0].keys(),
        )
        writer.writeheader()
        writer.writerows(plan_results_as_dict)
    logging.info("Saved results to " + str(results_file))

    if isinstance(logger, WandbLogger) and logger.experiment is not None:
        table_data = [
            list(plan_dict.values())  # dicts retain insertion order after 3.7
            for plan_dict in plan_results_as_dict
        ]
        logger.log_table(
            key=results_name,
            columns=list(plan_results_as_dict[0].keys()),
            data=table_data,
        )
        logger.finalize(status="success")


def test_lightning_agent_cli():
    # overwrite this because it might be set in the config.yaml.
    sys.argv.append("--data_layout.output_data.ensure_new_out_dir")
    sys.argv.append("false")
    cli = TestThundeRLCLI(run=False)
    lightning_adapter: LightningAdapter = cli.model
    in_data: InputData = cli.datamodule.data
    out_data = cli.config_init["data_layout.output_data"]
    test_setup: TestSetup = cli.config_init["test_setup"]
    test_lightning_agent(
        lightning_adapter=lightning_adapter,
        logger=cli.trainer.logger,
        input_data=in_data,
        output_data=out_data,
        test_setup=test_setup,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    test_lightning_agent_cli()
