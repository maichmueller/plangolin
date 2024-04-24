import tempfile
from test.fixtures import hetero_encoded_state, problem_setup

import pytest
from torch_geometric.loader import DataLoader

from rgnet.encoding.hetero_encoder import HeteroGraphEncoder
from rgnet.models.hetero_gnn import HeteroGNN
from rgnet.supervised.data import MultiInstanceSupervisedSet


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [["blocks", "small", "initial", 2], ["blocks", "small", "initial", 2]],
    indirect=True,
)
def test_hetero_gnn(hetero_encoded_state):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)

    model = HeteroGNN(
        hidden_size=encoder.hidden_size,
        num_layer=1,
        obj_name=encoder.obj_name,
        arity_by_pred=encoder.arity_by_pred,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


def test_hetero_batched():
    _, domain, problem = problem_setup("blocks", "small")
    encoder = HeteroGraphEncoder(domain, hidden_size=2)
    tmpdir: str = tempfile.mkdtemp()
    dataset = MultiInstanceSupervisedSet(
        [problem], encoder, force_reload=True, root=tmpdir
    )
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        model = HeteroGNN(
            hidden_size=2,
            num_layer=1,
            obj_name=encoder.obj_name,
            arity_by_pred=encoder.arity_by_pred,
        )

        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        assert out.size() == (batch.batch_size,)
