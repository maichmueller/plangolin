import logging
from collections import defaultdict
from test.fixtures import hetero_encoded_state, small_blocks

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest
import torch
from torch_geometric.data import HeteroData

import xmimir as xmi
from rgnet.encoding.hetero_encoder import HeteroGraphEncoder
from rgnet.utils import import_all_from


def test_hetero_data():
    domain, problems = import_all_from("test/pddl_instances/blocks")
    for prob in problems:
        if "large" in prob.filepath:
            continue
        logging.info("Testing problem: " + prob.name)
        state_space = xmi.XStateSpace.create(domain.filepath, prob.filepath)
        encoder = HeteroGraphEncoder(state_space.problem.domain)
        for state in state_space:
            data = encoder.to_pyg_data(encoder.encode(state))
            data.validate()
            validate_hetero_data(data, encoder)


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [["blocks", "small", "initial"], ["blocks", "small", "goal"]],
    indirect=True,
)
def test_decode(hetero_encoded_state):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)
    decoded = encoder.from_pyg_data(data)
    node_match = iso.categorical_node_match("type", None)
    edge_match = iso.numerical_edge_match("position", None)
    assert nx.is_isomorphic(
        graph, decoded, node_match=node_match, edge_match=edge_match
    )


def test_consistent_order_of_objects(small_blocks):
    """
    Its quite important that the order of objects in the torch_geometric encoding is consistent.
    Meaning that if object 'A' in encoder.to_pyg_data(encoder.encode(state)).x_dict["obj"] is at index i,
    then 'A' is also at index i for all other states of the same problem.
    """
    space, domain, medium_problem = small_blocks
    encoder = HeteroGraphEncoder(domain)
    initial = space.initial_state
    initial_pyg = encoder.to_pyg_data(encoder.encode(initial))

    def obj_to_on_g_edge_index(graph):
        return graph.get_edge_store(
            encoder.obj_type_id, "0", "on" + encoder.node_factory.goal_suffix
        ).edge_index

    obj_0_on_g_index: torch.Tensor = obj_to_on_g_edge_index(initial_pyg)

    successors = list(space.forward_transitions(initial))
    successors = [
        encoder.to_pyg_data(encoder.encode(target)) for _, target, _ in successors
    ]
    # We know which node 'a' is because the goal is on(a,b) so there should be one edge with attribute 0 from object-node a to atom-node on_g(a,b).
    successor_edge_indices = [obj_to_on_g_edge_index(g) for g in successors]
    assert all(
        (obj_0_on_g_index == successor_edge_index).all()
        for successor_edge_index in successor_edge_indices
    )


def validate_hetero_data(data: HeteroData, encoder: HeteroGraphEncoder):
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
            dest_indices = edge_index_dict[(encoder.obj_type_id, str(pos), node_type)][
                1
            ]
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
