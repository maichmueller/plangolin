from test.fixtures import fresh_flashdrive, hetero_encoded_state  # noqa: F401, F403

import pytest
import torch.nn.functional
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from plangolin.encoding import HeteroGraphEncoder
from plangolin.models.relational_gnn import RelationalGNN, ValueRelationalGNN
from plangolin.rl.data.flash_drive import attr_getters
from plangolin.utils.object_embeddings import ObjectPoolingModule
from xmimir import parse


@pytest.mark.parametrize(
    "hetero_encoded_state",
    [["blocks", "small", "initial"], ["blocks", "small", "initial"]],
    indirect=True,
)
def test_hetero_gnn(hetero_encoded_state):
    graph, encoder = hetero_encoded_state
    data = encoder.to_pyg_data(graph)

    model = ValueRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


@pytest.mark.parametrize(
    "fresh_flashdrive",
    [
        [
            "blocks",
            "small.pddl",
            dict(attribute_getters={"y": attr_getters.distance_to_goal}),
        ],
        [
            "blocks",
            "medium.pddl",
            dict(attribute_getters={"y": attr_getters.distance_to_goal}),
        ],
    ],
    indirect=["fresh_flashdrive"],  # only this one is a fixture
)
def test_value_hetero_gnn_backward(tmp_path, fresh_flashdrive):
    domain, problem = parse(fresh_flashdrive.domain_path, fresh_flashdrive.problem_path)
    encoder = HeteroGraphEncoder(domain)
    batch = Batch.from_data_list(fresh_flashdrive)
    model = ValueRelationalGNN(
        embedding_size=2,
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


@pytest.mark.parametrize(
    "fresh_flashdrive",
    [
        ["blocks", "small.pddl"],
    ],
    indirect=["fresh_flashdrive"],  # only this one is a fixture
)
def test_hetero_gnn_backward(tmp_path, fresh_flashdrive):
    """Calculate the embedding for each of the small-blocks states with one-hot encoding as target.
    This test makes sure that no in-place operations are used by the GNN that would obstruct backward-passes.
    """
    domain, problem = parse(fresh_flashdrive.domain_path, fresh_flashdrive.problem_path)
    encoder = HeteroGraphEncoder(domain)
    batch = Batch.from_data_list(fresh_flashdrive)
    assert len(batch) == 5, "If this fails the underlying test data has changed!"
    model = RelationalGNN(
        embedding_size=5,
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
        from plangolin.utils.object_embeddings import ObjectEmbedding

        embedding: ObjectEmbedding = ObjectEmbedding.from_sparse(*model(batch))
        out = pooling(embedding)
        target = torch.eye(n=5, dtype=torch.float)
        assert out.shape[0] == len(batch)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optim.step()


@pytest.mark.parametrize(
    "fresh_flashdrive",
    [
        ["blocks", "small.pddl"],
        ["blocks", "medium.pddl"],
    ],
    indirect=["fresh_flashdrive"],  # only this one is a fixture
)
def test_hetero_batched(tmp_path, fresh_flashdrive):
    domain, problem = parse(fresh_flashdrive.domain_path, fresh_flashdrive.problem_path)
    encoder = HeteroGraphEncoder(domain)
    loader = DataLoader(fresh_flashdrive, batch_size=3)
    for batch in loader:
        model = ValueRelationalGNN(
            embedding_size=2,
            num_layer=1,
            aggr="sum",
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )

        out = model(batch)

        assert out.size() == (batch.batch_size,)
