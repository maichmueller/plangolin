from test.fixtures import hetero_encoded_state
from test.supervised.test_data import create_dataset

import pytest
import torch.nn.functional
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from rgnet.models.hetero_gnn import HeteroGNN, ValueHeteroGNN
from rgnet.utils.object_embeddings import ObjectPoolingModule


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [["blocks", "small", "initial"], ["blocks", "small", "initial"]],
    indirect=True,
)
def test_hetero_gnn(hetero_encoded_state):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)

    model = ValueHeteroGNN(
        hidden_size=2,
        num_layer=1,
        aggr="sum",
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


def test_value_hetero_gnn_backward(tmp_path):
    dataset = create_dataset("small", tmp_path)
    batch = Batch.from_data_list(dataset)
    encoder = dataset.state_encoder
    model = ValueHeteroGNN(
        hidden_size=2,
        num_layer=1,
        aggr="sum",
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    model.train()
    optim = torch.optim.SGD(model.parameters(), 0.001)
    for i in range(3):
        optim.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        assert out.shape[0] == len(batch)
        loss = torch.nn.functional.mse_loss(out, batch.y.float())
        loss.backward()
        optim.step()


def test_hetero_gnn_backward(tmp_path):
    """Calculate the embedding for each of the small-blocks states with one-hot encoding as target.
    This test makes sure that no in-place operations are used by the GNN that would obstruct backward-passes.
    """
    dataset = create_dataset("small", tmp_path)
    batch = Batch.from_data_list(dataset)
    assert len(batch) == 5, "If this fails the underlying test data has changed!"
    encoder = dataset.state_encoder
    model = HeteroGNN(
        hidden_size=5,
        num_layer=1,
        aggr="sum",
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    model.train()
    optim = torch.optim.SGD(model.parameters(), 0.001)
    pooling = ObjectPoolingModule(pooling="add")
    for i in range(3):
        optim.zero_grad()
        from rgnet.utils.object_embeddings import ObjectEmbedding

        embedding: ObjectEmbedding = ObjectEmbedding.from_sparse(*model(batch))
        out = pooling(embedding)
        target = torch.eye(n=5, dtype=torch.float)
        assert out.shape[0] == len(batch)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optim.step()


def test_hetero_batched(tmp_path):
    dataset = create_dataset("small", tmp_path)
    encoder = dataset.state_encoder
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        model = ValueHeteroGNN(
            hidden_size=2,
            num_layer=1,
            aggr="sum",
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )

        out = model(batch)

        assert out.size() == (batch.batch_size,)
