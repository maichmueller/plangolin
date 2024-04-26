from test.encoding.test_color_encoder import problem_setup
from test.model.test_hetero_message_passing import get_enc_initial_and_goal

import mockito
import torch
import torch_geometric as pyg
from torch_geometric.data import Batch

from rgnet.encoding import HeteroEncoding
from rgnet.model import HeteroGNN


def test_hetero_gnn():
    space, domain, problem = problem_setup("blocks", "small")
    initial_state = space.get_initial_state()
    encoder = HeteroEncoding(domain, hidden_size=2)
    data = encoder.to_pyg_data(encoder.encode(initial_state))

    model = HeteroGNN(
        hidden_size=2,
        num_layer=1,
        obj_name=encoder.obj_name,
        arity_by_pred=encoder.arity_by_pred,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


def test_hetero_batched():

    hidden = 1

    def patched_forward(x):
        # We have to determine what output shape is expected
        if x.shape == (4, 2 * hidden):  # obj update, 4 because a1,b1,a2,b2
            return x[:, hidden:]  # just use the new emb

        # We have 2 graphs -> readout
        if x.shape == (2, hidden) and torch.allclose(
            x, torch.tensor([[3.0 + 6.0], [9.0 + 12.0]])
        ):
            return torch.tensor(-1.0)
        # identity function predicate-MLPs
        return x

    mockito.patch(pyg.nn.MLP.forward, patched_forward)

    enc, d_init, d_goal = get_enc_initial_and_goal(hidden_size=2)

    d_init[enc.obj_name].x = torch.tensor([[1.0], [2.0]])
    d_goal[enc.obj_name].x = torch.tensor([[3.0], [4.0]])
    batch = Batch.from_data_list([d_init, d_goal])
    model = HeteroGNN(
        hidden_size=hidden,
        num_layer=1,
        obj_name=enc.obj_name,
        arity_by_pred=enc.arity_by_pred,
    )
    out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
    assert torch.allclose(out, torch.tensor(-1.0))
