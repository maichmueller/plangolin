import csv
import datetime
import itertools
import logging
import multiprocessing
import queue
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Sequence

import torch
from lightning.pytorch.loggers import Logger, WandbLogger

import xmimir
from plangolin.encoding import EncoderFactory
from plangolin.logging_setup import get_logger, null_logger, tqdm
from plangolin.rl.data_layout import InputData, OutputData
from plangolin.rl.panorama import PanoramaCLI
from plangolin.rl.panorama.policy_gradient.cli import TestSetup
from plangolin.rl.panorama.utils import default_checkpoint_format, resolve_checkpoints
from plangolin.rl.reward import RewardFunction
from plangolin.rl.search.agent_maker import AgentMaker
from plangolin.rl.search.model_search import ModelSearch
from plangolin.utils.misc import DummyPbar, env_aware_cpu_count
from plangolin.utils.plan import Plan, ProbabilisticPlan
from plangolin.utils.system import exit_if_orphaned, increase_resource_limit
from xmimir import ActionHistoryDataPack, XProblem

logger = get_logger(__name__)

# Global worker identifier for pool workers
WORKER_ID: int | None = None


def _init_worker(worker_id_queue):
    global WORKER_ID
    WORKER_ID = worker_id_queue.get()
    threading.Thread(target=exit_if_orphaned, daemon=True).start()


def _listen_updates(q, bar):
    while True:
        try:
            elem = q.get(timeout=0.0)
            if isinstance(elem, tuple) and len(elem) == 3:
                epoch, step, prob_name = elem
                bar.update(1)
                bar.set_description(f"epoch={epoch} step={step} problem={prob_name}")
            elif isinstance(elem, StopIteration):
                bar.close()
                return
        except queue.Empty:
            pass


def process_result(
    checkpoint_path: Path,
    ckpt_results: list[ProbabilisticPlan],
    epoch: int,
    step: int,
    output_data: OutputData,
    cli_logger: Logger,
    logger_local: logging.Logger,
):
    logger_local.info(
        f"Finished checkpoint {checkpoint_path} (epoch={epoch}, step={step})"
    )
    solved = sum(p.solved for p in ckpt_results)
    logger_local.info(f"Solved {solved} out of {len(ckpt_results)}")
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
    logger_local.info("Saved results to " + str(results_file))

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
    logger_local.info(
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
    detail_level: int = 0,
    encoder_factory: EncoderFactory | None = None,
    test_instances: Sequence[XProblem] | None = None,
    domain_path: Path | None = None,
    problem_paths: Sequence[Path] | None = None,
    optimal_plans: Sequence[Plan | ActionHistoryDataPack] | None = None,
    update_queue: multiprocessing.queues.Queue | None = None,
    stop_flag: Any | None = None,
) -> tuple[Path, List[ProbabilisticPlan], int, int]:
    """
    Runs the rollout & analysis for every test problem for a single checkpoint.
    Returns: (checkpoint_path, results, epoch, step)
    NOTE: We pass the model's state_dict instead of the module itself to avoid pickling issues.
    """
    epoch, step = default_checkpoint_format(checkpoint_path.name)
    worker_id = WORKER_ID if WORKER_ID is not None else 0
    if detail_level == 0:
        logger_local = null_logger
    else:
        logger_local = get_logger(f"{__name__}.Worker({worker_id})")

    logger_local.info(f"Using checkpoint with epoch={epoch}, step={step}")

    test_instances = test_instances or []
    domain = None
    if not test_instances:
        if domain_path is None or problem_paths is None:
            raise ValueError(
                "No test instances provided and no domain or problem paths specified."
            )
        # Load test problems from the given domain and problem paths
        test_instances = []
        for problem_path in problem_paths:
            domain, problem = xmimir.parse(domain_path, problem_path)
            test_instances.append(problem)
    if encoder_factory is not None:
        # If an encoder factory is provided, use it to create the encoder. This will only happen if we go multi-process.
        encoder = encoder_factory(domain)
        agent_maker.encoder = encoder

    local_results: List[ProbabilisticPlan] = []
    for test_problem, optimal_plan in zip(
        test_instances, optimal_plans or itertools.repeat(None)
    ):
        # check for external stop signal
        if getattr(stop_flag, "value", False):
            raise Exception("Stop flag set, exiting worker")
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
        logger_local.info(
            f"Analyzed Results:\n{analyzed_data.str(detailed=(detail_level == 2))}:"
        )
        local_results.append(analyzed_data)

        if update_queue is not None:
            pass
            update_queue.put((epoch, step, test_problem.name), timeout=5)
    return checkpoint_path, local_results, epoch, step


def eval_model(
    cli: PanoramaCLI,
    num_workers: int = 0,
    progress_bar: bool = True,
    detail_level: int = 0,
    ckpt_filter: Sequence[str] | None = None,
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

    :param cli: The CLI instance used to run the agent.
    :param num_workers: Number of workers to use for parallel evaluation. If 0, runs in the main process.
    :param progress_bar: Whether to show a progress bar during evaluation.
    :param detail_level: Detail level for logging: 0 for minimal, 1 for basic, 2 for detailed output.
    :param ckpt_filter: Optional list of regex filters to apply to the checkpoints.
    """
    agent_maker: AgentMaker = cli.config_init.agent_maker
    input_data: InputData = cli.config_init.data_layout.input_data
    output_data: OutputData = cli.config_init.data_layout.output_data
    test_setup: TestSetup = cli.config_init.test_setup
    reward_function: RewardFunction = cli.config_init.reward
    cli_logger: WandbLogger = cli.trainer.logger
    encoder_factory: EncoderFactory = cli.config_init.encoder_factory

    if detail_level == 0:
        logger_local = null_logger
    else:
        logger_local = logger
    if not input_data.test_problems:
        raise ValueError("No test instances provided")

    checkpoints_paths, last_checkpoint = resolve_checkpoints(output_data)
    assert isinstance(checkpoints_paths, List)
    if len(checkpoints_paths) == 0:
        logger.warning("Provided an empty list as checkpoint_paths")
    assert all(
        isinstance(ckpt, Path) and ckpt.is_file() and ckpt.suffix == ".ckpt"
        for ckpt in checkpoints_paths
    )

    if ckpt_filter:
        # Filter checkpoints based on the provided regex patterns
        filtered_checkpoints = []
        for ckpt in checkpoints_paths:
            if any(re.search(pattern, ckpt.name) for pattern in ckpt_filter):
                filtered_checkpoints.append(ckpt)
        checkpoints_paths = filtered_checkpoints

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
    total_problems = len(input_data.test_problems) * len(checkpoints_paths)
    pbar = tqdm if progress_bar else DummyPbar
    pbar = pbar(
        range(total_problems),
        desc="Evaluating checkpoints",
        unit="problem",
        logger=logger,
    )
    if num_workers > 0:
        # shared stop flag for workers
        manager = multiprocessing.Manager()
        stop_flag = manager.Value("b", False)
        wid_queue = manager.Queue()  # Queue to communicate 1st the worker ids on init
        update_queue = manager.Queue()  # Queue for pbar updates to the main thread
        for wid in range(1, max_workers + 1):
            wid_queue.put(wid)

        # start listener thread to drain update_queue
        listener = threading.Thread(
            target=_listen_updates, args=(update_queue, pbar), daemon=True
        )
        listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(wid_queue,),
            ) as pool:
                agent_maker.encoder = None
                optimal_plans = [
                    (
                        ActionHistoryDataPack(t.action for t in plan.transitions)
                        if (plan := input_data.plan_by_problem.get(prob)) is not None
                        else None
                    )
                    for prob in input_data.test_problems
                ]

                for ckpt in checkpoints_paths:
                    futures.append(
                        pool.submit(
                            _evaluate_checkpoint,
                            ckpt,
                            model_search,
                            agent_maker,
                            encoder_factory=encoder_factory,
                            domain_path=Path(input_data.domain.filepath),
                            problem_paths=[
                                Path(prob.filepath) for prob in input_data.test_problems
                            ],
                            optimal_plans=optimal_plans,
                            test_setup=test_setup,
                            detail_level=detail_level,
                            update_queue=update_queue,
                            stop_flag=stop_flag,
                        )
                    )

                # collect as they finish
                for fut in as_completed(futures):
                    checkpoint_path, ckpt_results, epoch, step = fut.result()
                    test_results[(epoch, step)] = ckpt_results
                    metrics[(epoch, step)] = process_result(
                        checkpoint_path,
                        ckpt_results,
                        epoch,
                        step,
                        output_data,
                        cli_logger,
                        logger_local,
                    )
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user, stopping workers.")
            stop_flag.value = True
            for fut in futures:
                fut.cancel()
            sys.exit(1)
    else:
        # If no workers are specified, run in the main process
        for checkpoint_path in checkpoints_paths:
            _, ckpt_results, epoch, step = _evaluate_checkpoint(
                checkpoint_path,
                model_search,
                agent_maker,
                test_setup=test_setup,
                test_instances=input_data.test_problems,
                detail_level=detail_level,
                optimal_plans=[
                    input_data.plan_by_problem.get(prob)
                    for prob in input_data.test_problems
                ],
                update_queue=None,
            )
            test_results[(epoch, step)] = ckpt_results
            metrics[(epoch, step)] = process_result(
                checkpoint_path,
                ckpt_results,
                epoch,
                step,
                output_data,
                cli_logger,
                logger_local,
            )

    logger.info(
        "Metrics for all checkpoints:\n"
        + "\n".join(
            f"Checkpoint (epoch={epoch}, step={step}): "
            + ", ".join(f"{k}: {v}" for k, v in m.items())
            for (epoch, step), m in sorted(
                metrics.items(), key=lambda x: (x[0][0], x[0][1])
            )
        )
    )

    if isinstance(cli_logger, WandbLogger) and cli_logger.experiment is not None:
        cli_logger.finalize(status="success")


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # multiprocessing.set_start_method("fork", force=True)
    import argparse

    from jsonargparse._util import import_object

    increase_resource_limit()
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
    pre_parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Whether to show a progress bar during evaluation.",
    )
    pre_parser.add_argument(
        "--detail_level",
        type=int,
        default=0,
        help="Detail level for logging: 0 for minimal, 1 for basic, 2 for detailed output.",
    )

    def comma_list(s: str) -> list[str]:
        return [tok for tok in s.split(",") if tok]

    pre_parser.add_argument(
        "--ckpt_filter",
        type=comma_list,
        default=[],
        help="Comma-separated regex filters, e.g. '--cktp_filter=epoch=5,epoch=10'",
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
    cli_class = import_object(cli_name)
    cli = cli_class(run=False)
    logger.info(f"Running CLI: {cli_class!r}")
    eval_model(
        cli=cli,
        num_workers=known_args.num_workers,
        progress_bar=known_args.progress_bar,
        detail_level=known_args.detail_level,
        ckpt_filter=known_args.ckpt_filter,
    )
