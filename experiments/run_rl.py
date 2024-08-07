import argparse
import itertools
import logging
import pathlib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

import pymimir as mi
import torch
import torchrl
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch.nn import Parameter
from torch_geometric.nn import MLP
from torchrl.modules import ValueOperator
from torchrl.objectives import LossModule, ValueEstimators
from torchrl.objectives.value import TD0Estimator
from torchrl.trainers import Trainer

from rgnet import HeteroGraphEncoder
from rgnet.rl import Agent, EmbeddingModule, SimpleLoss
from rgnet.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.epsilon_greedy import EGreedyAgent
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
            current_embedding_key=Agent.default_keys.current_embedding,
            env=base_env,
            embedding_module=embedding_module,
        ),
        cache_specs=True,
        device=embedding_module.device,
    )
    return env


def calc_optimal_values(
    states, space: mi.StateSpace, gamma: float, device=torch.device("cpu")
):
    distances = torch.tensor(
        [space.get_distance_to_goal_state(s) for s in states],
        dtype=torch.int,
        device=device,
    )
    return -(1 - gamma**distances) / (1 - gamma)


class TD0Loss(torchrl.objectives.LossModule):
    def __init__(self, gamma, value_operator):
        super().__init__()
        self.value_operator = value_operator
        self.td0 = TD0Estimator(
            gamma=gamma,
            shifted=True,
            average_rewards=False,
            value_network=value_operator,
        )
        self.td0.set_keys(value=Agent.default_keys.state_value)

    def forward(self, tensordict: TensorDictBase):
        td = tensordict.clone(False)
        self.td0(td)
        estimates = self.value_operator(td)[Agent.default_keys.state_value].squeeze()
        targets = td[self.td0.value_target_key].squeeze()
        loss_out = torch.nn.functional.mse_loss(estimates, targets, reduction="none")
        return TensorDict(
            {
                "loss_critic": loss_out.mean(),
            }
        )


class SupervisedLoss(torchrl.objectives.LossModule):

    def __init__(self, value_op: ValueOperator, optimal_values: torch.Tensor):
        super().__init__()
        self.value_op = value_op
        self.optimal_values = optimal_values

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td = tensordict.clone(False)
        self.value_op(td)
        estimates = td[Agent.default_keys.state_value].squeeze()
        return TensorDict(
            {
                "loss_critic": torch.nn.functional.mse_loss(
                    estimates, self.optimal_values, reduction="mean"
                )
            }
        )


def simple_linear_net(hidden_size: int):
    return torch.nn.Linear(hidden_size, 1)


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


def validate(policy, value_op, env, space: mi.StateSpace, gamma, env_keys, atol=0.1):
    # --- Evaluate the agent --- #
    # NOTE that the embeddings are calculated with reset()
    #   Therefore, the eval_td should only be created after training!
    non_goal_states = [s for s in space.get_states() if not space.is_goal_state(s)]
    eval_td = env.reset(states=non_goal_states)
    optimal_values = calc_optimal_values(non_goal_states, space, gamma)
    predicted_values = value_op(eval_td).get(Agent.default_keys.state_value).squeeze(-1)
    logging.info(
        "--------------------------------Validation--------------------------------"
    )
    logging.info(f"{torch.nn.functional.l1_loss(optimal_values, predicted_values)=}")
    logging.info(f"{torch.allclose(optimal_values, predicted_values, atol=atol)=}")

    if policy is None:
        return
    # Optimal agent decisions
    best_successors = []
    for state in eval_td[env_keys.state]:
        targets = [t.target for t in space.get_forward_transitions(state)]
        best = min(targets, key=lambda s: space.get_distance_to_goal_state(s))
        best_successors.append(best)
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
        function()
    logging.info(prof.key_averages())


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
    non_goal_states = [s for s in states if not space.is_goal_state(s)]
    batch_size = batch_size or (num_states - 1)
    env = transformed_environment(space, embedding, batch_size=batch_size)

    return env, non_goal_states


def resolve_algorithm(
    parser_args, space, embedding, non_goal_states
) -> tuple[TensorDictModule | None, ValueOperator, LossModule, Iterator[Parameter]]:

    policy: TensorDictModule | None
    loss: LossModule
    value_op: ValueOperator
    optim_parameter: Iterator[torch.nn.Parameter] | List[dict[str, Any]]

    value_net = (
        mlp_net(embedding.hidden_size)
        if parser_args.value_net == "mlp"
        else simple_linear_net(embedding.hidden_size)
    )

    if parser_args.algorithm == "ac":
        agent = Agent(embedding, value_net=value_net)
        policy = agent.as_td_module(
            PlanningEnvironment.default_keys.state,
            PlanningEnvironment.default_keys.transitions,
            PlanningEnvironment.default_keys.action,
        )
        value_op = agent.value_operator
        loss = SimpleLoss(agent.value_operator)
        loss.make_value_estimator(ValueEstimators.TD0, gamma=parser_args.gamma)
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

    elif parser_args.algorithm == "egreedy":
        eagent = EGreedyAgent(
            value_net=value_net,
            embedding=embedding,
            eps_init=parser_args.egreedy_epsilon_init,
            eps_end=parser_args.egreedy_epsilon_end,
            annealing_num_steps=int(
                args.iterations * args.egreedy_epsilon_annealing_fraction
            ),
        )
        policy = eagent.as_td_module(
            ExpandedStateSpaceEnv.default_keys.transitions,
            ExpandedStateSpaceEnv.default_keys.action,
        )
        value_op = ValueOperator(
            value_net,
            in_keys=[Agent.default_keys.current_embedding],
            out_keys=[Agent.default_keys.state_value],
        )
        loss = TD0Loss(gamma=parser_args.gamma, value_operator=value_op)
        optim_parameter = eagent.parameters()

    elif parser_args.algorithm == "supervised":
        policy = None
        value_op = ValueOperator(
            value_net,
            in_keys=[Agent.default_keys.current_embedding],
            out_keys=[Agent.default_keys.state_value],
        )
        loss = SupervisedLoss(
            value_op, calc_optimal_values(non_goal_states, space, parser_args.gamma)
        )
        optim_parameter = itertools.chain(value_op.parameters(), embedding.parameters())
    else:
        raise ValueError(f"Unknown algorithm name {parser_args.algorithm}")

    return policy, value_op, loss, optim_parameter


def resolve_and_run(parser_args):
    logging.getLogger().setLevel(logging.INFO)
    space, domain, problem = resolve_problem(parser_args)

    device = (
        get_device_cuda_if_possible()
        if parser_args.device is None or parser_args.device == "auto"
        else torch.device(parser_args.device)
    )
    embedding_module = resolve_embedding(
        parser_args, space=space, domain=domain, device=device
    )

    env, non_goal_states = build_env(space, batch_size=None, embedding=embedding_module)

    policy, value_operator, loss_module, optim_params = resolve_algorithm(
        parser_args, space, embedding_module, non_goal_states
    )
    optimizer = torch.optim.Adam(optim_params, lr=parser_args.learning_rate)

    time_stamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    out_dir_root = parser_args.out_dir
    out_dir = pathlib.Path(out_dir_root) / parser_args.problem_name / time_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        space,
        non_goal_states,
        policy,
        value_operator,
        loss_module,
        optimizer,
        env,
        parser_args.gamma,
        parser_args.iterations,
        parser_args.atol,
        out_dir,
    )


def run(
    space,
    non_goal_states,
    policy,
    value_operator,
    loss_module,
    optimizer,
    env,
    gamma: float,
    iterations: int,
    atol: float,
    out_dir: pathlib.Path,
):

    trainer_file = out_dir / "trainer.pt"
    values_file = out_dir / "values.pt"

    log_num_parameter(optimizer=optimizer)

    def all_non_goal_states_reset(curr_env):
        return curr_env.reset(states=non_goal_states)

    trainer = torchrl.trainers.Trainer(
        collector=RolloutCollector(
            environment=env,
            policy=policy,
            rollout_length=1,
            custom_reset_func=all_non_goal_states_reset,
        ),
        loss_module=loss_module,
        optimizer=optimizer,
        frame_skip=1,
        total_frames=env.batch_size[0] * iterations,
        optim_steps_per_batch=1,
        progress_bar=True,
        save_trainer_file=trainer_file,
    )
    optimal_values = calc_optimal_values(
        states=non_goal_states, space=space, gamma=gamma, device=torch.device("cpu")
    )
    optimal_values_lookup = {s: v for s, v in zip(non_goal_states, optimal_values)}
    stopping_hook = ConsecutiveStopping(
        5,
        ValueFunctionConverged(
            value_operator=value_operator,
            reset_func=lambda: all_non_goal_states_reset(curr_env=env),
            optimal_values_lookup=optimal_values_lookup,
            atol=atol,
        ),
    )
    stopping_hook.register(trainer, "value_converged")

    logging_hook = LoggingHook(
        probs_key=Agent.default_keys.probs,
        action_key=PlanningEnvironment.default_keys.action,
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

    # Save the transition probabilities as nested tensor
    if logging_hook.probs_history:
        torch.save(
            [
                torch.nested.nested_tensor([t[0] for t in ls])
                for ls in logging_hook.probs_history
            ],
            out_dir / "probs.pt",
        )

    validate(policy, value_operator, env, space, gamma, env.keys)


if __name__ == "__main__":
    # Define default args
    DEFAULT_DEVICES = "auto"
    DEFAULT_OUT_PATH = "../out"
    DEFAULT_ITERATIONS = 2_000
    DEFAULT_ATOL = 0.1
    DEFAULT_GAMMA = 0.9
    DEFAULT_LEARNING_RATE = 0.02

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
        help="Only applies if algorithm =ac (Actor Critic)."
        "If set the parameters of the actor_net will be updated with a different"
        "learning rate then the general learning rate (--lr).",
    )
    parser.add_argument(
        "--egreedy_epsilon_init",
        type=float,
        required=False,
        default=0.5,
        help="Only applies for algorithm = egreedy. "
        "Initial epsilon value (default: 0.5)",
    )
    parser.add_argument(
        "--egreedy_epsilon_end",
        type=float,
        required=False,
        default=0.001,
        help="Only applies for algorithm = egreedy. "
        "Final epsilon value at the end of annealing (default: 0.001).",
    )
    parser.add_argument(
        "--egreedy_epsilon_annealing_fraction",
        type=float,
        required=False,
        default=0.8,
        help="Only applies for algorithm = egreedy. "
        "The fraction of iterations over which the epsilon value is annealed "
        "(default: 0.8).",
    )
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

    resolve_and_run(args)
