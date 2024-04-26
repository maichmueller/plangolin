import logging
from collections import defaultdict

import pymimir as mi
from torch_geometric.data import HeteroData

from rgnet.encoding.hetero_encoder import HeteroGraphEncoder
from rgnet.utils import import_all_from


def test_hetero_data():
    domain, problems = import_all_from("test/pddl_instances/blocks")
    encoder = HeteroGraphEncoder(domain, hidden_size=2)
    # problems = [problem, problem1, problem2]
    for prob in problems:
        if "large" in prob.name:
            continue
        logging.info("Testing problem: " + prob.name)
        state_space = mi.StateSpace.new(prob, mi.GroundedSuccessorGenerator(prob))
        for state in state_space.get_states():
            data = encoder.to_pyg_data(encoder.encode(state))
            data.validate()
            validate_hetero_data(data, encoder)


def validate_hetero_data(data: HeteroData, encoder: HeteroGraphEncoder):
    # for ["obj", *, pred], and ["obj", *, pred] there are exactly arity(pred) many edges
    assert encoder.obj_type in data.node_types
    x_dict = data.x_dict

    edge_index_dict = data.edge_index_dict

    for node_type in data.node_types:
        if node_type == encoder.obj_type:
            continue

        arity = encoder.arity_dict[node_type]

        allowed_atom_indices = set(range(x_dict[node_type].shape[0]))
        incoming_edges_by_atom = defaultdict(int)
        outgoing_edges_by_atom = defaultdict(int)
        for pos in range(arity):
            # Check that every atom has exactly arity many outgoing edges
            dest_indices = edge_index_dict[(encoder.obj_type, str(pos), node_type)][1]
            for dst_index in dest_indices:
                incoming_edges_by_atom[dst_index.item()] += 1
                assert dst_index.item() in allowed_atom_indices

            # Check that every atom has exactly arity many outgoing edges
            source_indices = edge_index_dict[(node_type, str(pos), encoder.obj_type)][0]
            for src_index in source_indices:
                outgoing_edges_by_atom[src_index.item()] += 1
                assert src_index.item() in allowed_atom_indices

        assert all(incoming_edges_by_atom[i] == arity for i in allowed_atom_indices)
        assert all(outgoing_edges_by_atom[i] == arity for i in allowed_atom_indices)
