import csv
import datetime
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence

import torch
from lightning.pytorch.loggers import Logger, WandbLogger

import xmimir
from rgnet.logging_setup import get_logger
from rgnet.rl.data_layout import InputData, OutputData
from rgnet.rl.reward import RewardFunction
from rgnet.rl.search.agent_maker import AgentMaker
from rgnet.rl.search.model_search import ModelSearch
from rgnet.rl.thundeRL import ThundeRLCLI
from rgnet.rl.thundeRL.policy_gradient.cli import TestSetup
from rgnet.rl.thundeRL.utils import default_checkpoint_format, resolve_checkpoints
from rgnet.utils.misc import env_aware_cpu_count
from rgnet.utils.plan import Plan, ProbabilisticPlan
from xmimir import ActionHistoryDataPack, XProblem


def eval_model(cli: ThundeRLCLI, num_workers: int = 0):
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

    :param cli: The CLI instance used to run the agent.
    """
    agent_maker: AgentMaker = cli.config_init.agent_maker
    input_data: InputData = cli.config_init.data_layout.input_data
    output_data: OutputData = cli.config_init.data_layout.output_data
    test_setup: TestSetup = cli.config_init.test_setup
    reward_function: RewardFunction = cli.config_init.reward
    cli_logger: WandbLogger = cli.trainer.logger

    if not input_data.test_problems:
        raise ValueError("No test instances provided")

    checkpoints_paths, last_checkpoint = resolve_checkpoints(output_data)
    assert isinstance(checkpoints_paths, List)
    if len(checkpoints_paths) == 0:
        get_logger(__name__).warning("Provided an empty list as checkpoint_paths")
    assert all(
        isinstance(ckpt, Path) and ckpt.is_file() and ckpt.suffix == ".ckpt"
        for ckpt in checkpoints_paths
    )

    # Determine device from Lightning CLI trainer
    try:
        device = cli.trainer.strategy.root_device
    except AttributeError:
        # Fallback to CPU if strategy unavailable
        device = torch.device("cpu")

    agent_maker.device = device

    model_search = ModelSearch(
        test_setup,
        reward_function=reward_function,
        device=device,
    )

    # Prepare data for pickling once
    max_workers = num_workers if num_workers > 0 else env_aware_cpu_count()
    futures = []
    test_results: dict[tuple[int, int], list[ProbabilisticPlan]] = dict()
    metrics: dict[tuple[int, int], dict[str, float]] = dict()
    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for ckpt in checkpoints_paths:
                futures.append(
                    pool.submit(
                        _evaluate_checkpoint,
                        ckpt,
                        model_search,
                        agent_maker,
                        domain_path=Path(input_data.domain.filepath),
                        problem_paths=[
                            Path(prob.filepath) for prob in input_data.test_problems
                        ],
                        test_setup=test_setup,
                        detailed=cli.config_init.detailed,
                    )
                )

            # collect as they finish
            for fut in as_completed(futures):
                checkpoint_path, ckpt_results, epoch, step = fut.result()
                test_results[(epoch, step)] = ckpt_results
                metrics[(epoch, step)] = process_result(
                    checkpoint_path, ckpt_results, epoch, step, cli_logger, output_data
                )
    else:
        # If no workers are specified, run in the main process
        for checkpoint_path in checkpoints_paths:
            _, ckpt_results, epoch, step = _evaluate_checkpoint(
                checkpoint_path,
                model_search,
                agent_maker,
                test_setup=test_setup,
                test_instances=input_data.test_problems,
                detailed=cli.config_init.detailed,
                optimal_plans=[
                    input_data.plan_by_problem.get(prob)
                    for prob in input_data.test_problems
                ],
            )
            test_results[(epoch, step)] = ckpt_results
            metrics[(epoch, step)] = process_result(
                checkpoint_path, ckpt_results, epoch, step, cli_logger, output_data
            )

    if isinstance(cli_logger, WandbLogger) and cli_logger.experiment is not None:
        cli_logger.finalize(status="success")


def process_result(
    checkpoint_path: Path,
    ckpt_results: list[ProbabilisticPlan],
    epoch: int,
    step: int,
    cli_logger: Logger,
    output_data: OutputData,
):
    logger = get_logger(__name__)
    logger.info(f"Finished checkpoint {checkpoint_path} (epoch={epoch}, step={step})")
    solved = sum(p.solved for p in ckpt_results)
    logger.info(f"Solved {solved} out of {len(ckpt_results)}")
    # Persist & (optionally) log each checkpoint's results immediately
    results_name = f"results_epoch={epoch}-step={step}"
    results_file = output_data.out_dir / (results_name + ".csv")
    plan_results_as_dict = [
        plan_result.serialize_as_dict() for plan_result in ckpt_results
    ]
    with open(results_file, "w") as f:
        writer = csv.DictWriter(
            f,
            plan_results_as_dict[0].keys(),
        )
        writer.writeheader()
        writer.writerows(plan_results_as_dict)
    logger.info("Saved results to " + str(results_file))

    solved_ckpt = sum(p.solved for p in ckpt_results)
    if not any(plan_result.optimal_transitions for plan_result in ckpt_results):
        plan_quality = "/"
    else:
        plan_quality = sum(
            len(plan_result.transitions) for plan_result in ckpt_results
        ) / sum(len(plan_result.optimal_transitions) for plan_result in ckpt_results)
    metrics = {
        "solved": solved_ckpt,
        "coverage": solved_ckpt / len(ckpt_results),
        "plan quality": plan_quality,
        "average rl return": sum(plan_result.rl_return for plan_result in ckpt_results)
        / len(ckpt_results),
        "average cost": sum(plan_result.cost for plan_result in ckpt_results)
        / len(ckpt_results),
        "average plan length": sum(
            len(plan_result.transitions) for plan_result in ckpt_results
        )
        / len(ckpt_results),
        "average subgoals": sum(plan_result.subgoals for plan_result in ckpt_results)
        / len(ckpt_results),
    }
    logger.info(
        f"Metrics for checkpoint {checkpoint_path} (epoch={epoch}, step={step}):\n"
        + "\n".join(f"{k}: {v}" for k, v in metrics.items())
    )
    if isinstance(cli_logger, WandbLogger) and cli_logger.experiment is not None:
        table_data = [list(plan_dict.values()) for plan_dict in plan_results_as_dict]
        cli_logger.log_table(
            key=results_name,
            columns=list(plan_results_as_dict[0].keys()),
            data=table_data,
            step=step,
        )

        cli_logger.log_metrics(
            metrics=metrics,
            step=step,
        )
    return metrics


def _evaluate_checkpoint(
    checkpoint_path: Path,
    model_search: ModelSearch,
    agent_maker: AgentMaker,
    test_setup: TestSetup,
    detailed: bool,
    test_instances: Sequence[XProblem] | None = None,
    domain_path: Path | None = None,
    problem_paths: Sequence[Path] | None = None,
    optimal_plans: Sequence[Plan | ActionHistoryDataPack] | None = None,
) -> tuple[Path, List[ProbabilisticPlan], int, int]:
    """
    Runs the rollout & analysis for every test problem for a single checkpoint.
    Returns: (checkpoint_path, results, epoch, step)
    NOTE: We pass the model's state_dict instead of the module itself to avoid pickling issues.
    """
    epoch, step = default_checkpoint_format(checkpoint_path.name)
    logger_local = get_logger(f"{__name__}.PID({os.getpid()})")
    logger_local.info(f"Using checkpoint with epoch={epoch}, step={step}")

    test_instances = test_instances or []
    if not test_instances:
        if domain_path is None or problem_paths is None:
            raise ValueError(
                "No test instances provided and no domain or problem paths specified."
            )
        # Load test problems from the given domain and problem paths
        test_instances = [
            xmimir.parse(domain_path, problem_path)[1] for problem_path in problem_paths
        ]
    local_results: List[ProbabilisticPlan] = []
    for test_problem, optimal_plan in zip(test_instances, optimal_plans or []):
        logger_local.info(
            f"Running rollout (max steps {test_setup.max_steps}) for problem {test_problem.name, test_problem.filepath}."
        )
        actor = agent_maker.agent(
            checkpoint_path=checkpoint_path,
            instance=test_problem,
            epoch=epoch,
        )
        successor_env = model_search.successor_env(test_problem)
        env = agent_maker.transformed_env(successor_env)
        start = time.time()
        rollout = model_search.rollout_on_env(env, actor)
        logger_local.info(
            f"Rollout completed in {datetime.timedelta(seconds=int(time.time() - start))}"
        )
        if optimal_plan is not None:
            if isinstance(optimal_plan, ActionHistoryDataPack):
                optimal_plan = Plan(
                    solved=True,
                    transitions=optimal_plan.reconstruct_sequence(
                        successor_env.active_instances[0]
                    ),
                    problem=test_problem,
                )
            assert isinstance(
                optimal_plan, Plan
            ), f"Expected optimal_plan to be a Plan or ActionHistoryDataPack, got {type(optimal_plan)}"

        analyzed_data: ProbabilisticPlan = model_search.analyze(
            test_problem,
            rollout,
            optimal_plan=optimal_plan,
        )
        logger_local.info(f"Analyzed Results:\n{analyzed_data.str(detailed=detailed)}:")
        local_results.append(analyzed_data)
    return checkpoint_path, local_results, epoch, step


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")

    import argparse

    from jsonargparse._util import import_object

    # --- Step 1: Pre-parse to extract --cli ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--cli", help="dot-path of the CLI to run (to be imported with jsonargparse)"
    )
    pre_parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for parallel evaluation. If 0, runs in the main process.",
    )
    known_args, remaining_args = pre_parser.parse_known_args()

    # # Remove the arguments from sys.argv so that the target CLI doesn't see it
    sys.argv = remaining_args

    sys.argv.extend(
        [
            # overwrite this because it might be set in the config.yaml.
            "--data_layout.output_data.ensure_new_out_dir",
            "false",
            # needs to be set to avoid loading all the training and validation data
            "--data.skip",
            "true",
            # Should be set to avoid overwriting the previous run with the same id (workaround because we can't set the default)
            "--trainer.logger.init_args.resume",
            "true",
        ]
    )
    cli_name = known_args.cli
    cli = import_object(cli_name)(run=False)
    eval_model(
        cli=cli,
    )
