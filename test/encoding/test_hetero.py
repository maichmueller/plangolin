import logging
from collections import defaultdict

import pymimir as mi
from torch_geometric.data import HeteroData

from rgnet.encoding import HeteroEncoding


def test_hetero_data():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/small.pddl").parse(domain)
    encoder = HeteroEncoding(domain, hidden_size=2)
    # problems = [problem, problem1, problem2]
    logging.info("Testing problem: " + problem.name)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    for state in state_space.get_states():
        data = encoder.to_pyg_data(encoder.encode(state))
        data.validate()
        validate_hetero_data(data, encoder)


def validate_hetero_data(data: HeteroData, encoder: HeteroEncoding):
    # for ["obj",*, pred], and ["obj",*, pred] there, are exactly arity(pred) many edges
    assert encoder.obj_name in data.node_types
    x_dict = data.x_dict

    edge_index_dict = data.edge_index_dict

    for pred_name in data.node_types:
        if pred_name == encoder.obj_name:
            continue

        arity = encoder.arity_by_pred[pred_name]

        allowed_atom_indices = set(range(x_dict[pred_name].shape[0]))
        incoming_edges_by_atom = defaultdict(int)
        outgoing_edges_by_atom = defaultdict(int)
        for pos in range(arity):
            # Check that every atom has exactly arity many outgoing edges
            dest_indices = edge_index_dict[(encoder.obj_name, str(pos), pred_name)][1]
            for dst_index in dest_indices:
                incoming_edges_by_atom[dst_index.item()] += 1
                assert dst_index.item() in allowed_atom_indices

            # Check that every atom has exactly arity many outgoing edges
            source_indices = edge_index_dict[(pred_name, str(pos), encoder.obj_name)][0]
            for src_index in source_indices:
                outgoing_edges_by_atom[src_index.item()] += 1
                assert src_index.item() in allowed_atom_indices

        assert all(incoming_edges_by_atom[i] == arity for i in allowed_atom_indices)
        assert all(outgoing_edges_by_atom[i] == arity for i in allowed_atom_indices)
