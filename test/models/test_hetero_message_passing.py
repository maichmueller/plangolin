from test.fixtures import problem_setup

import pytest
import torch
import torch_geometric as pyg
from mockito import arg_that, mock, when
from torch_geometric.data import Batch

from rgnet.encoding import HeteroGraphEncoder
from rgnet.models.hetero_message_passing import FanInMP, FanOutMP, SelectMP


def get_enc_initial_and_goal(hidden_size=1, problem: str = "small"):
    space, domain, problem = problem_setup("blocks", problem)
    encoder = HeteroGraphEncoder(domain, hidden_size)
    initial = space.get_initial_state()
    goal = space.get_goal_states()[0]

    d1 = encoder.to_pyg_data(encoder.encode(initial))
    d2 = encoder.to_pyg_data(encoder.encode(goal))
    return encoder, d1, d2


def test_fan_out():
    hidden = 1
    encoder, d1, d2 = get_enc_initial_and_goal(hidden_size=hidden)
    # Make objs clearly distinguishable
    d1[encoder.obj_type_id].x = torch.tensor([[1.0], [2.0]])
    d2[encoder.obj_type_id].x = torch.tensor([[3.0], [4.0]])
    batch: Batch = Batch.from_data_list([d1, d2])
    mlp_mock = mock(spec=pyg.nn.MLP)
    when(mlp_mock).__call__(arg_that(lambda x: x.shape == (2, 2 * hidden))).thenReturn(
        torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    )
    when(mlp_mock).__call__(arg_that(lambda x: x.shape == (1, 2 * hidden))).thenReturn(
        torch.tensor([[-3.0, -4.0]])
    )
    # clear(a),clear(b) in d1 and clear(a) in d2
    when(mlp_mock).__call__(
        arg_that(lambda x: x.shape == (3, hidden) and x[-1][-1] == 3.0)
    ).thenReturn(torch.tensor([[-1.0], [-1.0], [3.0]]))

    # ontable(a),ontable(b) in d1 and ontable(b) in d2
    when(mlp_mock).__call__(
        arg_that(lambda x: x.shape == (3, hidden) and x[-1][-1] == 4.0)
    ).thenReturn(torch.tensor([[-1.0], [-1.0], [4.0]]))

    mlps = dict()
    for pred, arity in encoder.arity_dict.items():
        mlps[pred] = mlp_mock

    fan_out = FanOutMP(mlps, encoder.obj_type_id)
    # Filter out empty edge_types (also includes handempty() and holding(x))
    edge_index_dict = {k: v for k, v in batch.edge_index_dict.items() if v.numel() != 0}
    out: dict[str, torch.Tensor] = fan_out(batch.x_dict, edge_index_dict)
    on_g_encoding = out["on" + encoder.node_factory.goal_suffix]
    ong1, ong2 = on_g_encoding[0], on_g_encoding[1]
    assert torch.allclose(ong1, torch.tensor([[-1.0, -2.0]]))
    assert torch.allclose(ong2, torch.tensor([[-3.0, -4.0]]))

    assert torch.allclose(out["clear"], torch.tensor([[-1.0], [-1.0], [3.0]]))
    assert torch.allclose(out["ontable"], torch.tensor([[-1.0], [-1.0], [4.0]]))
    assert encoder.obj_type_id not in out


def test_fan_in():
    hidden = 1
    encoder, d1, d2 = get_enc_initial_and_goal(hidden_size=hidden)
    # Make objs clearly distinguishable
    batch = Batch.from_data_list([d1, d2])
    batch["on" + encoder.node_factory.goal_suffix].x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]]
    )
    batch["on"].x = torch.tensor([[3.0, 4.0]])
    batch["clear"].x = torch.tensor([[1.0], [2.0], [3.0]])
    batch["ontable"].x = torch.tensor([[1.0], [2.0], [4.0]])

    fan_out = FanInMP(hidden, encoder.obj_type_id)
    # Filter out empty edge_types (also includes handempty() and holding(x))
    edge_index_dict = {k: v for k, v in batch.edge_index_dict.items() if v.numel() != 0}
    out: dict[str, torch.Tensor] = fan_out(batch.x_dict, edge_index_dict)
    assert len(out.keys()) == 1
    assert encoder.obj_type_id in out
    obj_embeddings = out[encoder.obj_type_id]
    assert obj_embeddings.shape == (4, hidden)
    a1, b1, a2, b2 = (
        obj_embeddings[0],
        obj_embeddings[1],
        obj_embeddings[2],
        obj_embeddings[3],
    )
    assert torch.allclose(a1, torch.tensor([1.0 + 1.0 + 1.0]))
    assert torch.allclose(b1, torch.tensor([2.0 + 2.0 + 2.0]))
    assert torch.allclose(a2, torch.tensor([3.0 + 3.0 + 3.0]))
    assert torch.allclose(b2, torch.tensor([4.0 + 4.0 + 4.0]))


def test_select_mp_hidden1():
    hidden = 1
    encoder, d1, d2 = get_enc_initial_and_goal(hidden_size=hidden)
    # Make objs clearly distinguishable
    batch = Batch.from_data_list([d1, d2])
    on_g = "on" + encoder.node_factory.goal_suffix
    batch[on_g].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch["on"].x = torch.tensor([[3.0, 4.0]])

    selector = SelectMP(hidden)
    obj = encoder.obj_type_id

    on_x = batch["on"].x, batch[obj].x
    on0_edge = batch["on", "0", obj].edge_index
    on1_edge = batch["on", "1", obj].edge_index
    ong_x = batch[on_g].x, batch[obj].x
    ong0_edge = batch[on_g, "0", obj].edge_index
    ong1_edge = batch[on_g, "1", obj].edge_index

    out_on_0 = selector(on_x, on0_edge, 0)
    assert torch.allclose(out_on_0, torch.tensor([[0.0], [0.0], [3.0], [0.0]]))

    out_on_1 = selector(on_x, on1_edge, 1)
    assert torch.allclose(out_on_1, torch.tensor([[0.0], [0.0], [0.0], [4.0]]))

    out_on_g_0 = selector(ong_x, ong0_edge, 0)
    assert torch.allclose(out_on_g_0, torch.tensor([[1.0], [0.0], [3.0], [0.0]]))
    out_on_g_1 = selector(ong_x, ong1_edge, 1)
    assert torch.allclose(out_on_g_1, torch.tensor([[0.0], [2.0], [0.0], [4.0]]))


def test_select_mp_hidden2():
    hidden = 2
    encoder, d1, d2 = get_enc_initial_and_goal(hidden_size=hidden)
    # Make objs clearly distinguishable
    batch = Batch.from_data_list([d1, d2])
    on_g = "on" + encoder.node_factory.goal_suffix
    batch[on_g].x = torch.tensor([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]])
    batch["on"].x = torch.tensor([[3.0, 3.0, 4.0, 4.0]])

    selector = SelectMP(hidden)
    obj = encoder.obj_type_id

    on_x = batch["on"].x, batch[obj].x
    on0_edge = batch["on", "0", obj].edge_index
    on1_edge = batch["on", "1", obj].edge_index
    ong_x = batch[on_g].x, batch[obj].x
    ong0_edge = batch[on_g, "0", obj].edge_index
    ong1_edge = batch[on_g, "1", obj].edge_index

    out_on_0 = selector(on_x, on0_edge, 0)
    assert torch.allclose(
        out_on_0, torch.tensor([[0.0] * 2, [0.0] * 2, [3.0] * 2, [0.0] * 2])
    )

    out_on_1 = selector(on_x, on1_edge, 1)
    assert torch.allclose(
        out_on_1, torch.tensor([[0.0] * 2, [0.0] * 2, [0.0] * 2, [4.0] * 2])
    )

    out_on_g_0 = selector(ong_x, ong0_edge, 0)
    assert torch.allclose(
        out_on_g_0, torch.tensor([[1.0] * 2, [0.0] * 2, [3.0] * 2, [0.0] * 2])
    )
    out_on_g_1 = selector(ong_x, ong1_edge, 1)
    assert torch.allclose(
        out_on_g_1, torch.tensor([[0.0] * 2, [2.0] * 2, [0.0] * 2, [4.0] * 2])
    )
