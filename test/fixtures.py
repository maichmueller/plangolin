import logging
import os
from pathlib import Path
from typing import List, Type

import mockito
import networkx as nx
import pytest
import torch
from matplotlib import pyplot as plt

import xmimir as xmi
from rgnet.encoding import ColorGraphEncoder, DirectGraphEncoder, HeteroGraphEncoder
from rgnet.encoding.base_encoder import EncoderFactory, GraphEncoderBase
from rgnet.rl.agents import ActorCritic
from rgnet.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from rgnet.rl.envs import ExpandedStateSpaceEnv, MultiInstanceStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, tolist
from rgnet.rl.thundeRL.flash_drive import FlashDrive
from rgnet.utils.object_embeddings import ObjectEmbedding


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


def problem_setup(
    domain_name, problem
) -> tuple[xmi.XStateSpace, xmi.XDomain, xmi.XProblem]:
    # Pycharm usually executes tests with .../test as working directory
    source_dir = "" if os.getcwd().endswith("/test") else "test/"
    domain_path = f"{source_dir}pddl_instances/{domain_name}/domain.pddl"
    problem_path = f"{source_dir}pddl_instances/{domain_name}/{problem}.pddl"
    space = xmi.XStateSpace.create(domain_path, problem_path)
    return space, space.problem.domain, space.problem


def request_accelerator_for_test(test_name: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.mps.is_available():
        return torch.device("mps:0")

    logging.warning(
        f"Tried to run device sensitive test {test_name} but accelerator was not available."
    )
    return torch.device("cpu")


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


def encoded_state(
    domain: str,
    problem: str,
    which_state: str,
    encoder_class: Type[GraphEncoderBase],
    **kwargs,
):
    space, domain, problem = problem_setup(domain, problem)

    if which_state == "initial":
        state = space.initial_state
    elif which_state == "goal":
        state = next(iter(space.goal_states_iter()))
    else:
        raise ValueError(
            "Unknown state wanted. Choose 'initial' or 'goal' state. Given: "
            + which_state
        )
    encoder = encoder_class(
        domain,
        **kwargs,
    )
    return encoder.encode(state), encoder


@pytest.fixture
def color_encoded_state(request):
    domain_param, prob_param, which_state_param, add_param = request.param
    return encoded_state(
        domain_param,
        prob_param,
        which_state_param,
        ColorGraphEncoder,
        enable_global_predicate_nodes=add_param,
    )


@pytest.fixture
def direct_encoded_state(request):
    domain_param, prob_param, which_state_param = request.param
    return encoded_state(
        domain_param,
        prob_param,
        which_state_param,
        DirectGraphEncoder,
    )


@pytest.fixture
def hetero_encoded_state(request):
    domain_param, prob_param, which_state_param = request.param
    return encoded_state(
        domain_param,
        prob_param,
        which_state_param,
        HeteroGraphEncoder,
    )


def random_object_embeddings(batch_size, num_object, hidden_size):
    dense_embeddings = torch.randn(
        size=(batch_size, num_object, hidden_size), requires_grad=True
    )
    padding_mask = torch.ones(size=(batch_size, num_object), dtype=torch.bool)
    return ObjectEmbedding(dense_embedding=dense_embeddings, padding_mask=padding_mask)


@pytest.fixture
def embedding_mock(hidden_size):

    def random_embeddings(states: List | NonTensorWrapper):
        states = tolist(states)
        batch_size = len(states)
        # shape is (batch_size, max_num_objects, hidden_size)
        return random_object_embeddings(batch_size, 4, hidden_size)

    empty_module = torch.nn.Module()
    empty_module.test_num_objects = 4
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
        encoder_factory=EncoderFactory(HeteroGraphEncoder),
    )
    return drive
