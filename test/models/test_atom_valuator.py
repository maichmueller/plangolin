from test.fixtures import (  # noqa: F401, F403
    hetero_encoded_state,
    medium_blocks,
    small_blocks,
)

import numpy as np
from torch_geometric.data import Batch

from rgnet.encoding import HeteroGraphEncoder
from rgnet.models.atom_valuator import AtomValuator
from rgnet.models.hetero_gnn import HeteroGNN


def test_atom_valuator(small_blocks, medium_blocks):
    space_small, domain_small, problem_small = small_blocks
    space_medium, domain_medium, problem_medium = medium_blocks
    encoder = HeteroGraphEncoder(domain_small)
    rng = np.random.default_rng(0)
    random_states = rng.choice(space_small, size=3).tolist()
    random_states += rng.choice(space_medium, size=4).tolist()
    rng.shuffle(random_states)

    encoded_states = [encoder.encode(state) for state in random_states]
    pyg_states = [encoder.to_pyg_data(graph) for graph in encoded_states]
    data = Batch.from_data_list(pyg_states)

    feature_size = 5
    gnn = HeteroGNN(
        embedding_size=feature_size,
        num_layer=1,
        aggr="sum",
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    valuator = AtomValuator(domain_small, feature_size=feature_size, assert_output=True)
    out, batch = gnn(data)
    value_dict, info = valuator(out, batch, data.object_names)
    assert len(value_dict) == len(domain_small.predicates())
    predicate_arities = list(valuator.arity_dict.values())
    num_objects = [len(s.problem.objects) for s in random_states]
    permutations_per_predicate = {
        pred: sum(obj_count**arity for obj_count in num_objects)
        for pred, arity in zip(valuator.arity_dict.keys(), predicate_arities)
    }

    for predicate in domain_small.predicates():
        predicate_name = predicate.name
        assert len(info[predicate_name]) == permutations_per_predicate[predicate_name]
        assert (
            value_dict[predicate_name].shape[0]
            == permutations_per_predicate[predicate_name]
        )
        assert value_dict[predicate_name].shape[1] == 1
