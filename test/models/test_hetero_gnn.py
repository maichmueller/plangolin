from test.fixtures import hetero_encoded_state
from test.supervised.test_data import create_dataset

import pytest
from torch_geometric.loader import DataLoader

from rgnet.models.hetero_gnn import HeteroGNN


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [["blocks", "small", "initial"], ["blocks", "small", "initial"]],
    indirect=True,
)
def test_hetero_gnn(hetero_encoded_state):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)

    model = HeteroGNN(
        hidden_size=2,
        num_layer=1,
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


def test_hetero_batched(tmp_path):
    dataset = create_dataset("small", tmp_path)
    encoder = dataset.encoder
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        model = HeteroGNN(
            hidden_size=2,
            num_layer=1,
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )

        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        assert out.size() == (batch.batch_size,)
