import argparse
import itertools
import logging
import pathlib
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pymimir as mi
import torch
import torchrl
from pymimir import State
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.nn import Parameter
from torch_geometric.nn import MLP
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ValueOperator
from torchrl.objectives import LossModule, ValueEstimators
from torchrl.record import WandbLogger
from torchrl.record.loggers import Logger as RLLogger

from experiments.analyze_rl_run import RLExperiment
from rgnet import HeteroGraphEncoder
from rgnet.rl import EmbeddingModule
from rgnet.rl.agents import (
    ActorCritic,
    EGreedyActorCriticHook,
    EGreedyModule,
    EpsilonAnnealing,
    OptimalPolicy,
    OptimalValueFunction,
    ValueModule,
)
from rgnet.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.losses import ActorCriticLoss, CriticLoss
from rgnet.rl.non_tensor_data_utils import non_tensor_to_list
from rgnet.rl.rollout_collector import RolloutCollector
from rgnet.rl.trainer_hooks import (
    ConsecutiveStopping,
    LoggingHook,
    ValueFunctionConverged,
)
from rgnet.utils import get_device_cuda_if_possible


def transformed_environment(space, embedding_module, batch_size):
    base_env = ExpandedStateSpaceEnv(space, batch_size=torch.Size((batch_size,)))
    env = NonTensorTransformedEnv(
        env=base_env,
        transform=EmbeddingTransform(
            current_embedding_key=ActorCritic.default_keys.current_embedding,
            env=base_env,
            embedding_module=embedding_module,
        ),
        cache_specs=True,
        device=embedding_module.device,
    )
    return env


def calc_optimal_values(
    states, space: mi.StateSpace, gamma: float, device=torch.device("cpu")
) -> torch.Tensor:
    distances = torch.tensor(
        [space.get_distance_to_goal_state(s) for s in states],
        dtype=torch.int,
        device=device,
    )
    return -(1 - gamma**distances) / (1 - gamma)


def simple_linear_net(hidden_size: int):
    return torch.nn.Linear(hidden_size, 1, bias=False)


def mlp_net(hidden_size: int):
    return MLP(
        channel_list=[hidden_size, hidden_size, 1],
        norm=None,
        dropout=0.0,
    )


def log_num_parameter(optimizer: torch.optim.Optimizer):
    num_params = sum(
        p.numel() for p in optimizer.param_groups[0]["params"] if p.requires_grad
    )
    logging.info(f"Training {num_params} parameter.")


def validate(
    policy, value_op, env, space: mi.StateSpace, optimal_values_dict, env_keys, atol=0.1
):
    # --- Evaluate the agent --- #
    # NOTE that the embeddings are calculated with reset()
    #   Therefore, the eval_td should only be created after training!
    eval_td = env.reset(states=space.get_states())
    predicted_values: torch.Tensor = (
        value_op(eval_td).get(ActorCritic.default_keys.state_value).squeeze(-1)
    )

    optimal_values = torch.tensor(
        [optimal_values_dict[s] for s in eval_td[env_keys.state]],
        device=predicted_values.device,
    )
    logging.info(
        "--------------------------------Validation--------------------------------"
    )
    logging.info(f"{torch.nn.functional.mse_loss(optimal_values, predicted_values)=}")
    logging.info(f"{torch.allclose(optimal_values, predicted_values, atol=atol)=}")

    if policy is None:
        return
    # Optimal agent decisions
    best_successors = []
    for state in eval_td[env_keys.state]:
        targets = [t.target for t in space.get_forward_transitions(state)]
        best = min(targets, key=lambda s: space.get_distance_to_goal_state(s))
        best_successors.append(best)
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        policy(eval_td)
    chosen_successors = [t.target for t in eval_td[env_keys.action]]
    matching_successors = sum(
        chosen_successors[i] == best_successors[i] for i in range(len(best_successors))
    )
    logging.info(
        f"Policy chose {matching_successors} out of {len(best_successors)} optimal successors"
    )


def with_profiler(function, use_gpu=True):
    with torch.profiler.profile(use_cuda=use_gpu, profile_memory=True) as prof:
        result = function()
    logging.info(prof.key_averages())
    return result


def one_hot_embedding(states, hidden_size: Optional[int] = None, device=None):
    hidden_size = hidden_size or len(states)
    num_states = len(states)
    device = device or torch.device("cpu")
    embedding: torch.Tensor
    if len(states) == hidden_size:
        embedding = torch.eye(
            hidden_size, hidden_size, dtype=torch.float, device=device
        )
    elif num_states > hidden_size:
        raise ValueError("Number of states must be less than or equal to hidden size")
    else:
        embedding = torch.nn.functional.one_hot(
            torch.arange(num_states, device=device), hidden_size
        ).float()

    class Embedding(torch.nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.lookup = {s: e for s, e in zip(states, embedding)}
            self.hidden_size = hidden_size
            self.device = device

        def forward(self, states_):
            states_ = non_tensor_to_list(states_)
            return torch.stack([self.lookup[s] for s in states_])

    return Embedding()


def resolve_problem(parser_args):
    domain_path = Path("../test/pddl_instances/blocks/domain.pddl")
    problem_path = Path(
        f"../test/pddl_instances/blocks/{parser_args.problem_name}.pddl"
    )
    domain = mi.DomainParser(str(domain_path.absolute())).parse()
    problem = mi.ProblemParser(str(problem_path.absolute())).parse(domain)
    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    return space, domain, problem


def resolve_embedding(parser_args, space, domain, device):
    if parser_args.embedding and parser_args.embedding_type == "gnn":
        num_layer = parser_args.gnn_layer
        if num_layer is None:
            num_layer = default_minimal_num_layer(parser_args.problem_name)
        hidden_size = parser_args.gnn_hidden_size
        aggr = parser_args.gnn_aggr
        return EmbeddingModule(
            HeteroGraphEncoder(domain),
            hidden_size=hidden_size,
            num_layer=num_layer,
            aggr=aggr,
            device=device,
        )
    # Otherwise use one-hot encoding
    assert parser_args.embedding_type is None or parser_args.embedding_type == "one_hot"
    if (
        hasattr(parser_args, "one_hot_hidden_size")
        and parser_args.one_hot_hidden_size is not None
    ):
        hidden_size = parser_args.one_hot_hidden_size
    else:
        hidden_size = len(space.get_states())
    return one_hot_embedding(space.get_states(), hidden_size=hidden_size, device=device)


def default_minimal_num_layer(problem_name):
    if problem_name == "small":
        return 1
    elif problem_name == "medium":
        return 3
    elif problem_name == "large":
        return 5
    else:
        raise ValueError(f"Unknown problem name {problem_name}")


def build_env(space, batch_size, embedding):
    states = space.get_states()
    logging.info(f"{len(states)=}")
    num_states = len(states)
    batch_size = batch_size or num_states
    env = transformed_environment(space, embedding, batch_size=batch_size)

    return env


def resolve_value_estimator(
    parser_args,
    loss: CriticLoss,
    device: torch.device,
    env_keys: PlanningEnvironment.AcceptedKeys,
    optimal_values_dict: Dict[mi.State, float],
):
    # shifted = True ->  the value and next value are estimated with a single call to the value network.
    # Also, important as otherwise torchrl uses vmap which breaks things.
    if parser_args.supervised_loss:
        ovf = OptimalValueFunction(optimal_values=optimal_values_dict, device=device)
        value_operator = ovf.as_td_module(
            state_key=env_keys.state,
            state_value_key=ActorCritic.default_keys.state_value,
        )
        loss.make_value_estimator(
            value_type=ValueEstimators.TD0,
            optimal_targets=value_operator,
            gamma=parser_args.gamma,
            shifted=True,
        )
    else:
        loss.make_value_estimator(
            ValueEstimators.TD0,
            gamma=parser_args.gamma,
            shifted=True,
        )


def resolve_egreedy(parser_args, embedding, value_net, env_keys):
    epsilon_annealing = EpsilonAnnealing.from_parser_args(parser_args)
    agent = ValueModule(
        embedding=embedding,
        value_net=value_net,
    )
    policy = TensorDictSequential(
        agent.as_td_module(env_keys.transitions, env_keys.action),
        EGreedyModule(
            epsilon_annealing,
            env_keys.transitions,
            env_keys.action,
            log_epsilon_actions=True,
        ),
    )
    policy.to(embedding.device)

    value_op = ValueOperator(
        value_net,
        in_keys=[ActorCritic.default_keys.current_embedding],
        out_keys=[ActorCritic.default_keys.state_value],
    )
    value_op.to(embedding.device)
    loss = CriticLoss(critic_network=value_op)
    loss.make_value_estimator(value_type=ValueEstimators.TD0, gamma=parser_args.gamma)
    loss.to(embedding.device)
    optim_parameter = agent.parameters()

    return policy, value_op, loss, optim_parameter


def resolve_actor_critic(parser_args, embedding, value_net, env_keys):
    agent = ActorCritic(embedding, value_net=value_net)
    agent.to(embedding.device)
    policy = agent.as_td_module(
        env_keys.state, env_keys.transitions, env_keys.action, add_probs=True
    )
    if parser_args.use_epsilon_for_actor_critic:
        epsilon_annealing = EpsilonAnnealing.from_parser_args(parser_args)
        policy = TensorDictSequential(
            policy,
            EGreedyModule(
                epsilon_annealing,
                env_keys.transitions,
                env_keys.action,
                log_epsilon_actions=True,
                replace_action_hook=EGreedyActorCriticHook(
                    agent.keys.probs, agent.keys.log_probs
                ),
            ),
        )
    value_op = agent.value_operator
    # includes policy, value_net and embeddings
    # use different learning rates for the policy and the value net
    if parser_args.separate_actor_loss:
        optim_parameter = [
            {
                "params": agent.actor_net.parameters(),
                "lr": parser_args.separate_actor_loss,
            },
            {"params": agent.value_operator.parameters()},
        ]
        if (emb_params := getattr(embedding, "parameters", None)) is not None:
            optim_parameter.append({"params": emb_params()})
    else:
        optim_parameter = agent.parameters()

    loss = ActorCriticLoss(
        agent.value_operator, log_prob_clip_value=parser_args.log_prob_clip_value
    )
    loss.to(embedding.device)

    return policy, value_op, loss, optim_parameter


def resolve_algorithm(
    parser_args,
    space,
    embedding,
    env_keys,
) -> tuple[
    TensorDictModule | None,
    ValueOperator,
    LossModule,
    Iterator[Parameter] | list[dict[str, Any]],
    dict[State, float],
]:
    policy: TensorDictModule | None
    loss: CriticLoss
    value_op: ValueOperator
    optim_parameter: Iterator[torch.nn.Parameter] | List[dict[str, Any]]

    value_net = (
        mlp_net(embedding.hidden_size)
        if parser_args.value_net == "mlp"
        else simple_linear_net(embedding.hidden_size)
    )

    if parser_args.algorithm == "ac":
        policy, value_op, loss, optim_parameter = resolve_actor_critic(
            parser_args, embedding, value_net, env_keys=env_keys
        )

    elif parser_args.algorithm == "egreedy":
        policy, value_op, loss, optim_parameter = resolve_egreedy(
            parser_args, embedding, value_net, env_keys=env_keys
        )

    elif parser_args.algorithm == "supervised":
        policy = OptimalPolicy(space=space).as_td_module(
            state_key=env_keys.state, action_key=env_keys.action
        )
        value_op = ValueOperator(
            value_net,
            in_keys=[ActorCritic.default_keys.current_embedding],
            out_keys=[ActorCritic.default_keys.state_value],
        )
        loss = CriticLoss(critic_network=value_op)
        optim_parameter = itertools.chain(value_op.parameters(), embedding.parameters())
        parser_args.supervised_loss = True  # make sure we use optimal values as targets
        value_op.to(embedding.device)
        loss.to(embedding.device)
    else:
        raise ValueError(f"Unknown algorithm name {parser_args.algorithm}")

    optimal_values = calc_optimal_values(
        states=space.get_states(),
        space=space,
        gamma=parser_args.gamma,
        device=embedding.device,
    )
    optimal_values_lookup: Dict[mi.State, float] = {
        s: v.item() for s, v in zip(space.get_states(), optimal_values)
    }

    resolve_value_estimator(
        parser_args=parser_args,
        loss=loss,
        device=embedding.device,
        env_keys=env_keys,
        optimal_values_dict=optimal_values_lookup,
    )
    return policy, value_op, loss, optim_parameter, optimal_values_lookup


def resolve_logger(parser_args, time_stamp: str, out_dir: pathlib.Path):
    experiment_name = f"{parser_args.problem_name}_{parser_args.algorithm}_{time_stamp}"
    logger = WandbLogger(
        exp_name=experiment_name,
        log_dir=out_dir,
        project="rgnet",
        group="rl",
    )
    logger.log_hparams(vars(parser_args))
    return logger


def resolve_and_run(parser_args):
    logging.getLogger().setLevel(logging.INFO)
    space, domain, problem = resolve_problem(parser_args)

    device = (
        get_device_cuda_if_possible()
        if parser_args.device is None or parser_args.device == "auto"
        else torch.device(parser_args.device)
    )
    logging.info(f"Running on: {device}")

    embedding_module = resolve_embedding(
        parser_args, space=space, domain=domain, device=device
    )

    env = build_env(space, batch_size=None, embedding=embedding_module)
    env_keys = env.base_env.keys

    policy, value_operator, loss_module, optim_params, optimal_values_lookup = (
        resolve_algorithm(parser_args, space, embedding_module, env_keys=env_keys)
    )
    optimizer = torch.optim.Adam(optim_params, lr=parser_args.learning_rate)

    time_stamp: str = datetime.now().strftime("%d-%m_%H-%M-%S")
    out_dir_root = parser_args.out_dir
    out_dir = pathlib.Path(out_dir_root) / parser_args.problem_name / time_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = resolve_logger(parser_args, time_stamp, out_dir)

    run(
        space,
        policy,
        value_operator,
        loss_module,
        optimizer,
        env,
        optimal_values_lookup,
        parser_args.iterations,
        parser_args.atol,
        out_dir,
        logger,
        log_interval=parser_args.log_interval,
    )
    return logger


def run(
    space,
    policy,
    value_operator,
    loss_module,
    optimizer,
    env,
    optimal_values_lookup,
    iterations: int,
    atol: float,
    out_dir: pathlib.Path,
    logger: RLLogger,
    log_interval: int,
):
    trainer_file = out_dir / "trainer.pt"
    values_file = out_dir / "values.pt"

    log_num_parameter(optimizer=optimizer)

    def all_states_reset(curr_env):
        return curr_env.reset(states=space.get_states())

    trainer = torchrl.trainers.Trainer(
        collector=RolloutCollector(
            environment=env,
            policy=policy,
            rollout_length=1,
            custom_reset_func=all_states_reset,
        ),
        loss_module=loss_module,
        optimizer=optimizer,
        frame_skip=1,
        total_frames=env.batch_size[0] * iterations,
        optim_steps_per_batch=1,
        progress_bar=True,
        save_trainer_file=trainer_file,
        logger=logger,
        log_interval=env.batch_size[0] * log_interval,
    )
    stopping_hook = ConsecutiveStopping(
        5,
        ValueFunctionConverged(
            value_operator=value_operator,
            reset_func=lambda: all_states_reset(curr_env=env),
            optimal_values_lookup=optimal_values_lookup,
            atol=atol,
        ),
    )
    stopping_hook.register(trainer, "value_converged")

    logging_hook = LoggingHook(
        logging_keys=[
            ActorCritic.default_keys.probs,
            PlanningEnvironment.default_keys.action,
            EGreedyModule.AcceptedKeys.epsilon_action_key,
        ]
    )
    logging_hook.register(trainer, "logging_hook")

    trainer.train()

    logging.info("Saving values under %s", values_file.absolute())
    values = torch.stack(stopping_hook.stopping_module.state_value_history)
    torch.save(values, values_file)

    for key, value in trainer._log_dict.items():
        if not key.startswith("loss_"):
            continue
        torch.save(torch.stack(value), out_dir / f"{key}.pt")
        logging.info("Saved %s", key)

    # Save the logged keys to disc
    # Done samples is number of encountered goal states per iteration List[float]
    torch.save(logging_hook.done_samples, out_dir / "done_samples.pt")

    # Selected actions
    actions: List[List[mi.Transition]] = logging_hook.logging_dict[
        PlanningEnvironment.default_keys.action
    ]
    forward_transitions = [space.get_forward_transitions(s) for s in space.get_states()]
    indices_of_actions: List[List[int]] = [
        [ft.index(action[0]) for action, ft in zip(batch_actions, forward_transitions)]
        for batch_actions in actions
    ]
    torch.save(indices_of_actions, out_dir / "actions.pt")

    if logging_hook.logging_dict[ActorCritic.default_keys.probs]:
        # Save as nested_tensor as non-uniform across batch and (potentially time)
        # List[nested_tensor[Tensor[batch_size x num_actions]]]
        torch.save(
            [
                torch.nested.nested_tensor([t[0] for t in ls])
                for ls in logging_hook.logging_dict[ActorCritic.default_keys.probs]
            ],
            out_dir / "probs.pt",
        )
    if logging_hook.logging_dict[EGreedyModule.default_keys.epsilon_action_key]:
        torch.save(
            torch.stack(
                logging_hook.logging_dict[EGreedyModule.default_keys.epsilon_action_key]
            ),
            out_dir / "epsilon.pt",
        )

    validate(policy, value_operator, env, space, optimal_values_lookup, env.keys)


def compute_animations(problem_name, exp_name, wlogger: Optional[WandbLogger] = None):
    exp = RLExperiment(
        blocks_instance=problem_name,
        run_name=exp_name,
        logger=wlogger,
    )
    exp.plot_graph_values_with_hist()
    exp.plot_graph_with_probs()
    logging.info("Saved plots under %s", exp.out_dir)


if __name__ == "__main__":
    # Define default args
    DEFAULT_DEVICES = "auto"
    DEFAULT_OUT_PATH = "../out"
    DEFAULT_ITERATIONS = 2_000
    DEFAULT_ATOL = 0.1
    DEFAULT_GAMMA = 0.9
    DEFAULT_LEARNING_RATE = 0.002
    DEFAULT_LOG_INTERVAL = 10

    parser = argparse.ArgumentParser(description="Process some modules.")
    parser.add_argument(
        "--problem",
        dest="problem_name",
        choices=["small", "medium", "large"],
        default="small",
    )
    parser.add_argument(
        "--algorithm", choices=["ac", "egreedy", "supervised"], default="ac"
    )
    parser.add_argument(
        "--separate_actor_loss",
        type=float,
        required=False,
        help="Only applies if algorithm = ac (Actor Critic)."
        "If set the parameters of the actor_net will be updated with a different"
        "learning rate then the general learning rate (--lr).",
    )
    parser.add_argument(
        "--supervised_loss",
        type=lambda x: bool(strtobool(str(x))),
        required=False,
        default=False,
    )
    parser.add_argument(
        "--log_prob_clip_value",
        type=float,
        required=False,
        help="Only applies if algorithm = ac (Actor Critic)."
        "If set the log probabilities will be clipped to the given value."
        "This can be especially useful if combined with --use_epsilon_for_actor_critic.",
    )
    parser.add_argument(
        "--use_epsilon_for_actor_critic",
        required=False,
        default=False,
        type=lambda x: bool(strtobool(str(x))),
        help="Only applies if algorithm = ac (Actor Critic). "
        "If set the epsilon greedy wrapper will be used around the actor. "
        "You can use --egreedy* arguments to specify the epsilon wrapper.",
    )
    EpsilonAnnealing.add_parser_args(parser)

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Discount factor for the value function. "
        f"Only required for ac and egreedy (default: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Total number of TD0 steps per batch dim (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--value_net",
        choices=["linear", "mlp"],
        default="mlp",
        help="Complexity of the value network (default: mlp)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICES,
        help=f"Device to use (default: {DEFAULT_DEVICES})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance for value function convergence",
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate for the optimizer (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_PATH,
        help=f"Output directory for the results (default: {DEFAULT_OUT_PATH})",
    )

    parser.add_argument(
        "--log_interval",
        default=DEFAULT_LOG_INTERVAL,
        type=int,
        help=f"Log interval in number of iterations,"
        f" that is independent of the batch size (default: {DEFAULT_LOG_INTERVAL})",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Use the HeteroGNN for embedding states if set, default uses one-hot encoding",
    )
    subparsers = parser.add_subparsers(
        dest="embedding_type", help="Choose the embedding type"
    )

    # Subparser for one-hot embedding (No additional arguments required)
    parser_one_hot = subparsers.add_parser("one_hot", help="Use one-hot embedding")
    parser_one_hot.add_argument("--one_hot_hidden_size", type=int, required=False)

    # Subparser for GNN embedding
    parser_gnn = subparsers.add_parser("gnn", help="Use GNN embedding")
    parser_gnn.add_argument(
        "--gnn_layer", type=int, required=False, help="Number of layers in the GNN"
    )
    parser_gnn.add_argument(
        "--gnn_hidden_size",
        type=int,
        required=True,
        help="Hidden size of the GNN layers",
    )
    parser_gnn.add_argument("--gnn_aggr", type=str, required=False, help="Aggregation")

    args = parser.parse_args()

    logger_ = resolve_and_run(args)

    compute_animations(args.problem_name, exp_name=logger_.exp_name, wlogger=logger_)
