import functools
import itertools
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mockito
import pymimir as mi
import torch
import torchrl
import tqdm
import wandb
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.nn import MLP
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators
from torchrl.objectives.value import TD0Estimator
from torchrl.trainers import Trainer

from rgnet import HeteroGraphEncoder
from rgnet.rl import Agent, EmbeddingModule, SimpleLoss
from rgnet.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.epsilon_greedy import EGreedyAgent
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack, non_tensor_to_list
from rgnet.rl.rollout_collector import RolloutCollector
from rgnet.utils import get_device_cuda_if_possible


def small_blocks():
    domain_path = Path("../test/pddl_instances/blocks/domain.pddl")
    problem_path = Path("../test/pddl_instances/blocks/small.pddl")
    domain = mi.DomainParser(str(domain_path.absolute())).parse()
    problem = mi.ProblemParser(str(problem_path.absolute())).parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


def medium_blocks():
    domain_path = Path("../test/pddl_instances/blocks/domain.pddl")
    problem_path = Path("../test/pddl_instances/blocks/medium.pddl")
    domain = mi.DomainParser(str(domain_path.absolute())).parse()
    problem = mi.ProblemParser(str(problem_path.absolute())).parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


def large_blocks():
    domain_path = Path("../test/pddl_instances/blocks/domain.pddl")
    problem_path = Path("../test/pddl_instances/blocks/large.pddl")
    domain = mi.DomainParser(str(domain_path.absolute())).parse()
    problem = mi.ProblemParser(str(problem_path.absolute())).parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


def normal_environment(space, batch_size):
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size((batch_size,)))


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


def embedding_generator(use_one_hot: bool = False, **kwargs):
    if use_one_hot:
        return lambda domain, states: one_hot_embedding(states, **kwargs)
    else:
        return lambda domain, states: embedding_module(domain, **kwargs)


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


def embedding_module(domain, hidden_size, num_layer, **kwargs):
    return EmbeddingModule(
        HeteroGraphEncoder(domain), hidden_size, num_layer, "sum", **kwargs
    )


def all_non_goal_states_reset_func(env, non_goal_states):
    def reset():
        return env.reset(states=non_goal_states)

    return reset


def random_initialization(env, states):
    def reset():
        # sample batch_size states from the states
        sampled_states = [
            states[i] for i in torch.randint(0, len(states), env.batch_size)
        ]
        return env.reset(states=sampled_states)

    return reset


def best_actions_func(space: mi.StateSpace):
    def step(transitions):
        transitions = non_tensor_to_list(transitions)
        best_actions = [
            min(ts, key=lambda t: space.get_distance_to_goal_state(t.target))
            for ts in transitions
        ]
        best_actions = as_non_tensor_stack(best_actions)
        return best_actions

    return TensorDictModule(
        module=step,
        in_keys=ExpandedStateSpaceEnv.default_keys.transitions,
        out_keys=ExpandedStateSpaceEnv.default_keys.action,
    )


def epsilon_greedy(value_net, embedding, total_steps=100):
    eagent = EGreedyAgent(
        value_net=value_net,
        embedding=embedding,
        eps_init=0.5,
        eps_end=0.0,
        annealing_num_steps=int(total_steps * 0.3),
    )
    return eagent.as_td_module(
        ExpandedStateSpaceEnv.default_keys.transitions,
        ExpandedStateSpaceEnv.default_keys.action,
    )


def fit(
    total_steps: Optional[int],
    reset_func,
    env,
    action_func,
    loss_func,
    optim,
    stop_early_condition=None,
    wandb_run=None,
):
    loss_history = []
    done_samples = 0
    if not total_steps and not stop_early_condition:
        raise ValueError("Either total steps or stopping condition required.")
    it = range(total_steps) if total_steps else itertools.count(start=0)
    with set_exploration_type(ExplorationType.RANDOM):
        for step in tqdm.tqdm(it, desc="Training Progress", total=total_steps):
            # if step % 100 == 0:
            #    print(f"{torch.cuda.memory_summary(device=None, abbreviated=False)=}")
            #    print(f"{torch.cuda.memory_allocated()=}")
            optim.zero_grad()
            td = reset_func()
            td = env.rollout(1, policy=action_func, auto_reset=False, tensordict=td)
            done_samples += torch.count_nonzero(td[("next", "done")]).item()
            if wandb_run:
                wandb_run.log({"done_samples": done_samples})
            loss = loss_func(td)
            loss_history.append(loss.item())
            if wandb_run:
                wandb_run.log({"loss": loss_history[-1]})
            loss.backward()
            optim.step()

            if stop_early_condition is not None and stop_early_condition(loss=loss):
                print("Stopped early at time step ", step)
                break

    print("Sampled ", done_samples, " in ", step, " total_steps")
    return loss_history


def calc_optimal_values(
    states, space: mi.StateSpace, gamma: float, device=torch.device("cpu")
):
    distances = torch.tensor(
        [space.get_distance_to_goal_state(s) for s in states],
        dtype=torch.int,
        device=device,
    )
    return -(1 - gamma**distances) / (1 - gamma)


def supervised_loss_func(value_op: ValueOperator, target: torch.Tensor):
    def loss(td):
        estimates = value_op(td)[Agent.default_keys.state_value].squeeze()
        return torch.nn.functional.mse_loss(estimates, target, reduction="mean")

    return loss


def td0_loss_func(gamma, value_operator, wandb_run=None):
    td0 = TD0Estimator(
        gamma=gamma,
        shifted=True,
        average_rewards=False,
        value_network=value_operator,
    )
    td0.set_keys(value=Agent.default_keys.state_value)

    def loss(td):
        td0(td)
        estimates = value_operator(td)[Agent.default_keys.state_value].squeeze()
        targets = td[td0.value_target_key].squeeze()
        loss_out = torch.nn.functional.mse_loss(estimates, targets, reduction="none")
        if wandb_run:
            wandb_run.log(
                {f"loss_{i}": loss_out[i].cpu().item() for i in range(len(loss_out))}
            )
        return loss_out.mean()

    return loss


def simple_linear_net(hidden_size: int):
    return torch.nn.Linear(hidden_size, 1)


def mlp_net(hidden_size: int):
    return MLP(
        channel_list=[hidden_size, hidden_size, 1],
        norm=None,
        dropout=0.0,
    )


def print_num_parameter(opimizer):
    print(
        "Training ",
        sum(p.numel() for p in opimizer.param_groups[0]["params"] if p.requires_grad),
        " parameter",
    )


class ConsecutiveStopping:

    def __init__(self, times: int, stopping_module):
        self.stopping_module = stopping_module
        self.times = times
        self.counter = 0

    def __call__(self, *args, **kwargs):
        if self.stopping_module(*args, **kwargs):
            self.counter += 1
            return self.counter >= self.times
        else:
            self.counter = 0
            return False


class LossSmallerThreshold:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, loss: torch.Tensor, **kwargs):
        return loss.cpu().item() < self.threshold


class ValueFunctionConverged:
    def __init__(
        self, reset_func, value_operator, optimal_values: torch.Tensor, atol=0.01
    ):
        self.counter = 0
        self.reset_func = reset_func
        self.value_operator = value_operator
        self.optimal_values = optimal_values
        self.atol = atol
        self.state_values = []

    def __call__(self, *args, **kwargs):
        td = self.reset_func()
        with torch.no_grad():
            self.value_operator.eval()
            predicted_values: torch.Tensor = (
                self.value_operator(td).get(Agent.default_keys.state_value).squeeze(-1)
            )
            self.state_values.append(predicted_values.detach().cpu())
            self.value_operator.train()
            return torch.allclose(self.optimal_values, predicted_values, atol=self.atol)


def validate(policy, value_op, env, space: mi.StateSpace, gamma, env_keys, atol=0.1):
    # --- Evaluate the agent --- #
    # NOTE that the embeddings are calculated with reset()
    #   Therefore, the eval_td should only be created after training!
    non_goal_states = [s for s in space.get_states() if not space.is_goal_state(s)]
    eval_td = env.reset(states=non_goal_states)
    optimal_values = calc_optimal_values(non_goal_states, space, gamma)
    predicted_values = value_op(eval_td).get(Agent.default_keys.state_value).squeeze(-1)
    print(f"{torch.nn.functional.l1_loss(optimal_values, predicted_values)=}")
    print(f"{torch.allclose(optimal_values, predicted_values, atol=atol)=}")

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
    print(
        f"Policy chose {matching_successors} out of {len(best_successors)} optimal successors"
    )


def env_setup(blocks, embedding_func, batch_size=None):
    space, domain, _ = blocks
    states = space.get_states()
    print(f"{len(states)=}")
    num_states = len(states)
    non_goal_states = [s for s in states if not space.is_goal_state(s)]
    embedding = embedding_func(domain=domain, states=states)
    batch_size = batch_size or (num_states - 1)
    env = transformed_environment(space, embedding, batch_size=batch_size)

    return space, env, embedding, non_goal_states


def supervised(
    blocks=small_blocks(),
    value_net: torch.nn.Module | None = None,
    total_steps: int = None,
    embedding_func=embedding_generator(True),
    gamma: float = 0.9,
    atol=0.1,
    device=torch.device("cpu"),
    out_dir=None,
):
    space, env, embedding, non_goal_states = env_setup(blocks, embedding_func)
    if value_net is None:
        value_net = mlp_net(embedding.hidden_size)

    optim = torch.optim.Adam(
        value_net.parameters(),
        # itertools.chain(value_net.parameters(), embedding.parameters()),
        0.001,
        weight_decay=0.0001,
    )
    print_num_parameter(optim)
    value_operator = ValueOperator(
        value_net,
        in_keys=[Agent.default_keys.current_embedding],
    )
    # embedding.train()
    value_net.train()
    reset_func = all_non_goal_states_reset_func(env, non_goal_states)

    # Optimal value function
    optimal_values_non_goal = calc_optimal_values(non_goal_states, space, gamma)
    stopping_criteria = ConsecutiveStopping(
        times=5,
        stopping_module=ValueFunctionConverged(
            reset_func, value_operator, optimal_values_non_goal, atol=atol
        ),
    )

    fit(
        total_steps=total_steps,
        reset_func=reset_func,
        env=env,
        action_func=best_actions_func(space),
        loss_func=supervised_loss_func(value_operator, optimal_values_non_goal),
        optim=optim,
        stop_early_condition=stopping_criteria,
    )
    # --- Evaluate the agent --- #
    if out_dir:
        predicted_values = torch.stack(stopping_criteria.stopping_module.state_values)
        diffs = predicted_values - optimal_values_non_goal.cpu()
        torch.save(diffs, out_dir + "values.pt")
        torch.save(value_net.state_dict(), out_dir + "value_net.pt")
        print("Saved values.pt and value_net.pt to ", out_dir)

    validate(
        best_actions_func(space), value_operator, env, space, gamma, env.default_keys
    )


def egredy(
    blocks=small_blocks(),
    value_net: torch.nn.Module | None = None,
    total_steps: int = 10_000,
    embedding_func=embedding_generator(
        use_one_hot=True, hidden_size=8, device=torch.device("cpu")
    ),
    gamma: float = 0.9,
    device: torch.device = torch.device("cpu"),
    out_dir=None,
):
    space, env, embedding, non_goal_states = env_setup(
        blocks,
        embedding_func,
    )

    env_keys = ExpandedStateSpaceEnv.default_keys
    if value_net is None:
        value_net = mlp_net(embedding.hidden_size)
    agent = EGreedyAgent(
        value_net=value_net,
        embedding=embedding,
        eps_init=0.5,
        eps_end=0.0,
        annealing_num_steps=int(0.7 * total_steps),
    )
    # wandb.watch(embedding.gnn)
    # wandb.watch(value_net)
    value_operator = ValueOperator(
        value_net,
        in_keys=[Agent.default_keys.current_embedding],
        out_keys=[Agent.default_keys.state_value],
    )
    egredy_module = agent.as_td_module(
        ExpandedStateSpaceEnv.default_keys.transitions,
        ExpandedStateSpaceEnv.default_keys.action,
    )
    optim = torch.optim.Adam(
        agent.parameters(),
        0.001,
        weight_decay=0.0001,
    )
    print_num_parameter(optim)

    reset_func = all_non_goal_states_reset_func(env, non_goal_states)

    optimal_values = calc_optimal_values(non_goal_states, space, gamma, device=device)

    value_stopping = ValueFunctionConverged(reset_func, value_operator, optimal_values)
    fit(
        total_steps=total_steps,
        reset_func=reset_func,
        env=env,
        action_func=egredy_module,  # best_actions_func(base_td, space),
        loss_func=td0_loss_func(gamma, value_operator),
        optim=optim,
        stop_early_condition=ConsecutiveStopping(
            times=5, stopping_module=value_stopping
        ),
        # wandb_run=run,
    )
    predicted_values = torch.stack(value_stopping.state_values)
    diffs = predicted_values - optimal_values.cpu()
    if out_dir is not None:
        torch.save(diffs, out_dir + "values.pt")
        torch.save(agent.state_dict(), out_dir + "agent.pt")
        print("Saved values.pt and agent.pt to ", out_dir)
    # --- Evaluate the agent --- #
    validate(egredy_module, value_operator, env, space, gamma, env_keys)


def ac_setup(embedding, env_keys, device, gamma):
    agent = Agent(embedding)
    agent.to(device)

    policy = agent.as_td_module(env_keys.state, env_keys.transitions, env_keys.action)

    loss = SimpleLoss(agent.value_operator, reduction="mean")
    loss.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    loss.to(device)

    optim = torch.optim.Adam(agent.parameters(), 0.001, weight_decay=0.0001)
    print_num_parameter(optim)

    def combined_loss(td_):
        loss_out = loss(td_)
        return loss_out["loss_critic"] + loss_out["loss_actor"]

    return agent, policy, loss, optim, combined_loss


def actor_critic(
    blocks=small_blocks(),
    total_steps=30_000,
    gamma=0.9,
    embedding_func=embedding_generator(False, hidden_size=8, num_layer=1),
    batch_size=None,
    device=get_device_cuda_if_possible(),
    out_dir=None,
):
    space, env, embedding, non_goal_states = env_setup(
        blocks,
        embedding_func,
        batch_size=batch_size,
    )
    env_keys = ExpandedStateSpaceEnv.default_keys

    agent, policy, loss, optim, combined_loss = ac_setup(
        embedding, env_keys, device, gamma
    )
    reset_func = all_non_goal_states_reset_func(env, non_goal_states)

    # Optimal value function
    optimal_values = calc_optimal_values(non_goal_states, space, gamma, device=device)
    stopping_criteria = ConsecutiveStopping(
        times=5,
        stopping_module=ValueFunctionConverged(
            reset_func, agent.value_operator, optimal_values, atol=0.1
        ),
    )
    fit(
        total_steps=total_steps,
        reset_func=reset_func,
        env=env,
        action_func=policy,  # best_actions_func(base_td, space),
        loss_func=combined_loss,
        optim=optim,
        stop_early_condition=stopping_criteria,
    )

    predicted_values = torch.stack(stopping_criteria.stopping_module.state_values)
    diffs = predicted_values - optimal_values.cpu()
    if out_dir is not None:
        torch.save(diffs, out_dir + "values.pt")
        torch.save(agent.state_dict(), out_dir + "agent.pt")
        print("Saved values.pt and agent.pt to ", out_dir)

    validate(policy, agent.value_operator, env, space, gamma, env_keys)


def gpu_embedding(blocks=small_blocks()):
    device = get_device_cuda_if_possible()
    space, domain, _ = blocks
    states = space.get_states()

    num_states = len(states)
    embedding = embedding_module(domain, num_states, device=device)
    embedding.to(device)

    base_env = ExpandedStateSpaceEnv(space, batch_size=torch.Size((4,)))
    reset_func = all_non_goal_states_reset_func(base_env, space)
    td = reset_func()
    embeddings: torch.Tensor = embedding(td[ExpandedStateSpaceEnv.default_keys.state])
    print(embeddings.device)
    # assert embeddings.device == device
    td[Agent.default_keys.current_embedding] = embeddings
    # one batch entry per non-goal state
    td["gpu"] = torch.ones((4, 1), device=device)


def with_trainer(
    blocks,
    total_steps=10,
    num_layer=1,
    gamma=0.9,
    device=get_device_cuda_if_possible(),
):

    space, env, embedding, non_goal_states = env_setup(
        blocks,
        lambda domain, states: one_hot_embedding(states),
        # domain, num_layer=num_layer, hidden_size=8, device=device
        # ),
    )
    env_keys = ExpandedStateSpaceEnv.default_keys

    agent, policy, loss, optim, combined_loss = ac_setup(
        embedding, env_keys, device, gamma
    )
    mockito.spy2(env._reset)
    mockito.spy2(env._step)
    mockito.spy2(loss.forward)
    mockito.spy2(optim.step)
    mockito.spy2(policy.forward)
    mockito.spy2(embedding.forward)

    data_loader = RolloutCollector(
        env,
        policy,
        rollout_length=1,
        exploration_type=torchrl.envs.utils.ExplorationType.RANDOM,
    )

    trainer = Trainer(
        collector=data_loader,
        total_frames=total_steps * env.batch_size[0],
        loss_module=loss,
        optimizer=optim,
        optim_steps_per_batch=1,
        frame_skip=1,
        progress_bar=True,
    )
    trainer.train()

    mockito.verify(env, times=total_steps)._reset(...)
    mockito.verify(env, times=total_steps)._step(...)
    mockito.verify(loss, times=total_steps).forward(...)
    mockito.verify(optim, times=total_steps).step(...)
    mockito.verify(policy, times=total_steps).forward(...)
    # embeddings gets called by _reset, _step and by the agent (for successor-state embeddings)
    mockito.verify(embedding, times=total_steps * 3).forward(...)

    validate(policy, agent.value_operator, env, space, gamma, env_keys)


def with_profiler(function, use_gpu=True):
    with torch.profiler.profile(use_cuda=use_gpu, profile_memory=True) as prof:
        function()
    print(prof.key_averages())


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        problem_name = "small"
    else:
        problem_name = argv[1]

    if problem_name == "small":
        problem = small_blocks()
        num_layer_ = 1
    elif problem_name == "medium":
        problem = medium_blocks()
        num_layer_ = 3
    elif problem_name == "large":
        problem = large_blocks()
        num_layer_ = 5
    else:
        raise ValueError(f"Unknown problem name {problem_name}")

    time_stamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    out_dir_ = f"/work/rleap1/jakob.krude/projects/remote/rgnet/out/{problem_name}/{time_stamp}/"
    pathlib.Path(out_dir_).mkdir(parents=True, exist_ok=True)

    device_ = get_device_cuda_if_possible()

    embedding_kwargs = {"hidden_size": 8, "num_layer": num_layer_, "device": device_}

    embedding_func_ = embedding_generator(False, **embedding_kwargs)

    total_steps_ = 4_000

    if len(argv) == 3:
        algorithm_name = argv[2]
        if algorithm_name == "ac":
            actor_critic(
                blocks=problem,
                embedding_func=embedding_func_,
                total_steps=total_steps_,
                device=device_,
                out_dir=out_dir_,
            )
        elif algorithm_name == "egreedy":
            egredy(
                blocks=problem,
                embedding_func=embedding_func_,
                total_steps=total_steps_,
                gamma=0.9,
                device=device_,
                out_dir=out_dir_,
            )
        elif algorithm_name == "supervised":
            supervised(
                blocks=problem,
                embedding_func=embedding_func_,
                total_steps=total_steps_,
                device=device_,
                out_dir=out_dir_,
            )
        else:
            raise ValueError(f"Unknown algorithm name {algorithm_name}")
