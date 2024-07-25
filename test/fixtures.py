import os
from typing import List

import mockito
import networkx as nx
import pymimir as mi
import pytest
import torch
from matplotlib import pyplot as plt

from rgnet import ColorGraphEncoder, DirectGraphEncoder, HeteroGraphEncoder
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list


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
