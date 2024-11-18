import logging
import os
from pathlib import Path
from typing import List

import mockito
import networkx as nx
import pymimir as mi
import pytest
import torch
from matplotlib import pyplot as plt

from rgnet.encoding import ColorGraphEncoder, DirectGraphEncoder, HeteroGraphEncoder
from rgnet.rl import ActorCritic
from rgnet.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from rgnet.rl.envs import ExpandedStateSpaceEnv, MultiInstanceStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list
from rgnet.rl.thundeRL.flash_drive import FlashDrive


def _draw_networkx_graph(graph: nx.Graph, **kwargs):
    nx.draw_networkx(
        graph,
        with_labels=kwargs.get("with_labels", True),
        labels=kwargs.get("labels", {n: str(n) for n in graph.nodes}),
        nodelist=kwargs.get("nodelist", [n for n in graph.nodes]),
        node_color=kwargs.get(
            "node_color", [attr["feature"] for _, attr in graph.nodes.data()]
        ),
        cmap=kwargs.get("cmap", "tab10"),
        **kwargs,
    )
    plt.show()


def problem_setup(domain_name, problem):
    # Pycharm usually executes tests with .../test as working directory
    source_dir = "" if os.getcwd().endswith("/test") else "test/"
    domain = mi.DomainParser(
        f"{source_dir}pddl_instances/{domain_name}/domain.pddl"
    ).parse()
    problem = mi.ProblemParser(
        f"{source_dir}pddl_instances/{domain_name}/{problem}.pddl"
    ).parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


def request_cuda_for_test(test_name: str) -> torch.device:
    if not torch.cuda.is_available():
        logging.warning(
            f"Tried to run device sensitive test {test_name} but cuda was not available."
        )
        return torch.device("cpu")
    return torch.device("cuda:0")


# Use a shared blocks-problem, neither of space, domain or problem should be mutated.
@pytest.fixture(scope="session")
def small_blocks():
    return problem_setup("blocks", "small")


@pytest.fixture(scope="session")
def medium_blocks():
    return problem_setup("blocks", "medium")


@pytest.fixture(scope="session")
def large_blocks():
    return problem_setup("blocks", "large")


@pytest.fixture
def color_encoded_state(request):
    domain_param, prob_param, which_state_param, add_param = request.param
    space, domain, _ = problem_setup(domain_param, prob_param)
    if which_state_param == "initial":
        state = space.get_initial_state()
    elif which_state_param == "goal":
        state = space.get_goal_states()[0]
    else:
        raise ValueError(
            "Unknown state wanted. Choose initial or goal state. Given: "
            + which_state_param
        )
    encoder = ColorGraphEncoder(domain, add_global_predicate_nodes=add_param)
    return (
        encoder.encode(state),
        encoder,
    )


@pytest.fixture
def direct_encoded_state(request):
    domain_param, prob_param, which_state_param = request.param
    space, domain, _ = problem_setup(domain_param, prob_param)
    if which_state_param == "initial":
        state = space.get_initial_state()
    else:
        state = space.get_goal_states()[0]
    encoder = DirectGraphEncoder(domain)
    return encoder.encode(state), encoder


@pytest.fixture
def hetero_encoded_state(request):
    domain_param, prob_param, which_state_param = request.param
    space, domain, _ = problem_setup(domain_param, prob_param)
    if which_state_param == "initial":
        state = space.get_initial_state()
    else:
        state = space.get_goal_states()[0]
    encoder = HeteroGraphEncoder(domain)
    return (
        encoder.encode(state),
        encoder,
    )


@pytest.fixture
def embedding_mock(hidden_size):

    def random_embeddings(states: List | NonTensorWrapper):
        states = non_tensor_to_list(states)
        batch_size = len(states)
        return torch.randn(size=(batch_size, hidden_size), requires_grad=True)

    empty_module = torch.nn.Module()
    empty_module.device = torch.device("cpu")
    mockito.when(empty_module).forward(...).thenAnswer(random_embeddings)
    empty_module.hidden_size = hidden_size
    return empty_module


@pytest.fixture
def multi_instance_env(request, spaces=None, batch_size=None):
    if spaces is None or batch_size is None:
        blocks_names, batch_size = request.param
        blocks = [request.getfixturevalue(name) for name in blocks_names]
        spaces = [space for space, _, _ in blocks]
    return MultiInstanceStateSpaceEnv(
        spaces=spaces,
        batch_size=torch.Size((batch_size,)),
        seed=42,
    )


@pytest.fixture
def expanded_state_space_env(request, blocks=None, batch_size=None):
    if blocks is None or batch_size is None:
        blocks_name, batch_size = request.param
        blocks = request.getfixturevalue(blocks_name)

    space, domain, _ = blocks
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size((batch_size,)), seed=42)


@pytest.fixture
def transformed_env(request, environment=None, embedding=None):
    if environment is None or embedding is None:
        env_fixture, embedding_fixture = request.param
        # environment has to be requested before with the correct arguments
        environment = request.getfixturevalue(env_fixture)
        embedding = request.getfixturevalue(embedding_fixture)
    return NonTensorTransformedEnv(
        env=environment,
        transform=EmbeddingTransform(
            current_embedding_key=ActorCritic.default_keys.current_embedding,
            env=environment,
            embedding_module=embedding,
        ),
    )


@pytest.fixture
def fresh_drive(tmp_path, force_reload=True):
    source_dir = Path("" if os.getcwd().endswith("/test") else "test/")
    data_dir = source_dir / "pddl_instances" / "blocks"
    problem_path = data_dir / "medium.pddl"
    domain_path = data_dir / "domain.pddl"
    drive = FlashDrive(
        problem_path=problem_path,
        domain_path=domain_path,
        custom_dead_end_reward=-100.0,
        root_dir=str(tmp_path.absolute()),
        force_reload=force_reload,
    )
    return drive
