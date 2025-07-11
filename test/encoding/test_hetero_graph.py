import itertools
from collections import defaultdict
from test.fixtures import (  # noqa: F401
    hetero_encoded_state,
    medium_blocks,
    small_blocks,
)

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest
import torch
from torch_geometric.data import HeteroData

import xmimir as xmi
from rgnet.encoding.hetero_encoder import HeteroGraphEncoder
from rgnet.encoding.ilg_hetero_encoder import HeteroILGGraphEncoder
from rgnet.logging_setup import get_logger
from rgnet.utils import import_all_from


@pytest.mark.parametrize(
    "encoder_type",
    [HeteroGraphEncoder, HeteroILGGraphEncoder],
    ids=["hetero", "hetero_ilg"],
)
@pytest.mark.parametrize("domain", ["blocks", "blocks_eq", "spanner"])
def test_hetero_data(domain, encoder_type):
    domain_name, problems = import_all_from(f"test/pddl_instances/{domain}")
    for prob in problems:
        if "large" in prob.filepath:
            continue
        get_logger(__name__).info(f"testing {prob.name} with {encoder_type.__name__!r}")
        state_space = xmi.XStateSpace(prob)
        encoder = encoder_type(state_space.problem.domain)

        for state in state_space:
            data = encoder.to_pyg_data(encoder.encode(state))
            data.validate()
            validate_hetero_data(data, encoder)


@pytest.mark.parametrize("domain", ["blocks", "blocks_eq", "spanner"])
def test_hetero_sat_goal_encoding(domain):
    domain_name, problems = import_all_from(f"test/pddl_instances/{domain}")
    for prob in problems:
        if "large" in prob.filepath:
            continue
        state_space = xmi.XStateSpace(prob)
        encoder = HeteroGraphEncoder(
            state_space.problem.domain, add_goal_satisfied_atoms=True
        )

        goals = state_space.problem.goal()
        for state in state_space:
            sat_goals = tuple(state.satisfied_literals(goals))
            graph = encoder.encode(state)
            enc_sat_nodes = list(
                filter(lambda n: n.endswith(encoder.goal_satisfied_suffix), graph.nodes)
            )
            assert len(enc_sat_nodes) == len(sat_goals)
            sat_atom_edges = defaultdict(list)
            for node in enc_sat_nodes:
                for atom, obj in graph.edges(node):
                    sat_atom_edges[atom].append(obj)
            for goal in sat_goals:
                gatom = goal.atom
                objs = set(o.get_name() for o in gatom.objects)
                enc_atom_str = (
                    encoder.node_factory(goal) + encoder.goal_satisfied_suffix
                )
                enc_objs = sat_atom_edges[enc_atom_str]
                assert len(enc_objs) == len(objs), (
                    f"Encoded objects {enc_objs} for goal {goal} "
                    f"do not match expected objects {objs}."
                )
                for enc_obj in enc_objs:
                    assert enc_obj in objs, (
                        f"Encoded object {enc_obj} for goal {goal} "
                        f"not in expected objects {objs}."
                    )


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [
        a + b
        for a, b in itertools.product(
            [
                ["blocks", "small", "initial"],
                ["blocks", "small", "goal"],
                ["blocks", "medium", "initial"],
                ["blocks", "medium", "goal"],
                ["blocks_eq", "small", "initial"],
                ["blocks_eq", "small", "goal"],
                ["blocks_eq", "medium", "initial"],
                ["blocks_eq", "medium", "goal"],
            ],
            [
                [{"add_goal_satisfied_atoms": False}],
                [{"add_goal_satisfied_atoms": True}],
            ],
        )
    ],
    indirect=True,
    ids=[
        f"{a}-{b}"
        for a, b in itertools.product(
            [
                "blocks-small-initial",
                "blocks-small-goal",
                "blocks-medium-initial",
                "blocks-medium-goal",
                "blocks_eq-small-initial",
                "blocks_eq-small-goal",
                "blocks_eq-medium-initial",
                "blocks_eq-medium-goal",
            ],
            [
                "no_goal_satisfied_atoms",
                "goal_satisfied_atoms",
            ],
        )
    ],
)
@pytest.mark.parametrize(
    ["encoder_type", "node_attrs", "node_defaults"],
    list(
        zip(
            [HeteroGraphEncoder, HeteroILGGraphEncoder],
            [["type"], ["type", "status"]],
            [[None], [None, None]],
        )
    ),
    ids=[
        "HeteroGraphEncoder",
        "HeteroILGGraphEncoder",
    ],
)
def test_decode(
    hetero_encoded_state,
    encoder_type,
    node_attrs,
    node_defaults,
):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)
    decoded = encoder.from_pyg_data(data)
    node_match = iso.categorical_node_match(node_attrs, node_defaults)
    edge_match = iso.numerical_multiedge_match(["position"], [None])
    assert nx.is_isomorphic(
        graph, decoded, node_match=node_match, edge_match=edge_match
    )


@pytest.mark.parametrize(
    "encoder_type",
    [HeteroGraphEncoder, HeteroILGGraphEncoder],
    ids=["HeteroGraphEncoder", "HeteroILGGraphEncoder"],
)
def test_consistent_order_of_objects(encoder_type, small_blocks):
    """
    Its quite important that the order of objects in the torch_geometric encoding is consistent.
    Meaning that if object 'A' in encoder.to_pyg_data(encoder.encode(state)).x_dict["obj"] is at index i,
    then 'A' is also at index i for all other states of the same problem.
    """
    space, domain, medium_problem = small_blocks
    encoder = encoder_type(domain)
    initial = space.initial_state
    initial_pyg = encoder.to_pyg_data(encoder.encode(initial))

    def obj_to_on_g_edge_index(graph):
        if isinstance(encoder, HeteroILGGraphEncoder):
            # we dont separate on and on_g in ILG encodings, so just use "on"
            # (which we know is in the graph due to the goal having an "on" atom)
            return graph.get_edge_store(encoder.obj_type_id, "0", "on").edge_index
        else:
            return graph.get_edge_store(
                encoder.obj_type_id, "0", "on" + encoder.node_factory.goal_suffix
            ).edge_index

    obj_0_on_g_index: torch.Tensor = obj_to_on_g_edge_index(initial_pyg)

    successors = list(space.forward_transitions(initial))
    successors = [
        encoder.to_pyg_data(encoder.encode(target)) for _, target, _ in successors
    ]
    # We know which node `a` is because the goal is `on(a,b)`,
    # so there should be one edge with attribute 0 from object-node `a` to atom-node `on_g(a,b)`.
    successor_edge_indices = [obj_to_on_g_edge_index(g) for g in successors]
    assert all(
        (obj_0_on_g_index == successor_edge_index).all()
        for successor_edge_index in successor_edge_indices
    )


@pytest.mark.parametrize(
    "encoder_type",
    [HeteroGraphEncoder, HeteroILGGraphEncoder],
    ids=["HeteroGraphEncoder", "HeteroILGGraphEncoder"],
)
def test_consistent_object_node_to_names(encoder_type, small_blocks, medium_blocks):
    space, domain, medium_problem = small_blocks
    space2, domain2, medium_problem2 = medium_blocks
    encoder = encoder_type(domain)
    initial = space.initial_state
    initial_pyg = encoder.to_pyg_data(encoder.encode(initial))
    successors = [
        encoder.to_pyg_data(encoder.encode(target))
        for _, target, _ in space.forward_transitions(initial)
    ]
    initial = space2.initial_state
    initial_pyg2 = encoder.to_pyg_data(encoder.encode(initial))
    successors2 = [
        encoder.to_pyg_data(encoder.encode(target))
        for _, target, _ in space2.forward_transitions(initial)
    ]
    assert all(
        (initial_pyg.object_names == successor.object_names) for successor in successors
    )
    assert all(
        initial_pyg.object_names == successor.object_names for successor in successors
    )
    assert all(
        (initial_pyg2.object_names == successor2.object_names)
        for successor2 in successors2
    )
    assert initial_pyg.object_names != initial_pyg2.object_names and any(
        successor.object_names != successor2.object_names
        for successor in successors
        for successor2 in successors2
    )


def validate_hetero_data(
    data: HeteroData, encoder: HeteroILGGraphEncoder | HeteroGraphEncoder
):
    # for ["obj", *, pred], and ["obj", *, pred] there are exactly arity(pred) many edges
    assert encoder.obj_type_id in data.node_types
    x_dict = data.x_dict

    edge_index_dict = data.edge_index_dict

    for node_type in data.node_types:
        if node_type == encoder.obj_type_id:
            continue

        arity = encoder.arity_dict[node_type]

        allowed_atom_indices = set(range(x_dict[node_type].shape[0]))
        incoming_edges_by_atom = defaultdict(int)
        outgoing_edges_by_atom = defaultdict(int)
        for pos in range(arity):
            # Check that every atom has exactly arity many outgoing edges
            dest_indices = edge_index_dict[
                (
                    encoder.obj_type_id,
                    str(pos),
                    node_type,
                )
            ][1]
            for dst_index in dest_indices:
                incoming_edges_by_atom[dst_index.item()] += 1
                assert dst_index.item() in allowed_atom_indices

            # Check that every atom has exactly arity many outgoing edges
            source_indices = edge_index_dict[
                (node_type, str(pos), encoder.obj_type_id)
            ][0]
            for src_index in source_indices:
                outgoing_edges_by_atom[src_index.item()] += 1
                assert src_index.item() in allowed_atom_indices

        assert all(incoming_edges_by_atom[i] == arity for i in allowed_atom_indices)
        assert all(outgoing_edges_by_atom[i] == arity for i in allowed_atom_indices)
