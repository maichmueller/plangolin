from test.fixtures import fresh_flashdrive, medium_blocks  # noqa: F401

import mockito
import torch.optim
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators

from rgnet.models import HeteroGNN, PyGHeteroModule
from rgnet.rl.agents import ActorCritic
from rgnet.rl.losses import ActorCriticLoss
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.policy_gradient_lit_module import PolicyGradientLitModule
from rgnet.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule


def test_training_step(fresh_flashdrive, medium_blocks):
    """
    Integration test for the training step of the PolicyGradientModule.
    Mocked: HeteroGNN, ActorCritic, ValueOperator
    Tested: ActorCriticLoss, PolicyGradientModule
    Tests he exact loss of PolicyGradientModule::training_step.
    Setup:
    We use a fixed batch size of five states.
    The done and reward signals are fixed such that only the first state is terminal.
    The embeddings of the current state are arranged from 0...num_objects_per_state.sum().
    The embeddings of the successors are fixed such that after pooling all are equal to one.
    The value function just forwards the pooled embeddings.
    Not really important what the actor does as all successors are equal.

    :param fresh_drive: initialized flash drive for medium blocks.
    """

    BATCH_SIZE = 5
    GAMMA = 0.9

    batch = [fresh_flashdrive[i] for i in range(BATCH_SIZE)]
    batch[0].done = torch.full_like(batch[0].done, fill_value=True)
    batch[0].reward = torch.full_like(batch[0].reward, fill_value=0.0)
    batched_tuple = collate_fn(batch)
    real_num_successors: torch.Tensor = batched_tuple[2]
    total_successors = real_num_successors.sum().item()

    # The first state only has three objects, the other have four.
    num_objects_per_state = torch.tensor([3] + [4] * (BATCH_SIZE - 1), dtype=torch.long)
    # Enumerate states and objects with embedding size = 1
    # Shape [num_objects_per_state.sum(), 1]
    current_embeddings_flat = torch.arange(
        num_objects_per_state.sum().item(), dtype=torch.float, requires_grad=True
    ).unsqueeze(dim=-1)
    # The batch indices to current_embeddings_flat.
    current_embeddings_batch = torch.arange(BATCH_SIZE).repeat_interleave(
        num_objects_per_state
    )
    current_embeddings = (current_embeddings_flat, current_embeddings_batch)
    current_object_embeddings = ObjectEmbedding.from_sparse(*current_embeddings)
    assert current_object_embeddings.dense_embedding.requires_grad
    # last object of first state is fake
    assert not current_object_embeddings.padding_mask[0, -1]

    # Successors have the same number of objects as source state.
    total_num_successors_objects: int = num_objects_per_state.dot(
        real_num_successors
    ).item()
    # Construct successors such that after object-pooling all are equal to one.
    successor_embeddings_flat = torch.cat(
        [
            torch.full(
                size=((real_num_successors[i] * num_objects_per_state[i]).item(),),
                fill_value=1 / num_objects_per_state[i].item(),
            )
            for i in range(BATCH_SIZE)
        ]
    ).unsqueeze(dim=-1)
    # successors of first state have 3 objects all others have 4
    successors_batch = torch.cat(
        [
            torch.arange(0, real_num_successors[0].item()).repeat_interleave(3),
            torch.arange(
                real_num_successors[0].item(), total_successors
            ).repeat_interleave(4),
        ]
    ).long()
    assert successors_batch.numel() == total_num_successors_objects
    assert successors_batch.max().item() == total_successors - 1

    successor_embeddings = (successor_embeddings_flat, successors_batch)
    successor_object_embeddings = ObjectEmbedding.from_sparse(*successor_embeddings)
    assert torch.allclose(
        ObjectPoolingModule(pooling="add")(successor_object_embeddings),
        torch.ones(size=(total_successors,)),
    )

    # Expected values for current_states is sum aggregation of current_embeddings
    # shape [batch_size, 1]
    expected_current_values = current_object_embeddings.dense_embedding.nansum(
        dim=1
    ).squeeze()

    def gnn_forward(batch):
        x_dict, edge_index_dict, batch_dict = PyGHeteroModule.unpack(batch)
        batch_size = batch_dict["obj"].max().item() + 1
        if batch_size == BATCH_SIZE:
            return current_embeddings
        return successor_embeddings

    gnn_mock = mockito.mock(HeteroGNN)
    mockito.when(gnn_mock).__call__(...).thenAnswer(gnn_forward)

    value_net_mock = ObjectPoolingModule(pooling="add")

    mockito.spy2(value_net_mock.forward)

    operator_mock = ValueOperator(
        value_net_mock,
        [ActorCritic.default_keys.current_embedding],
        [ActorCritic.default_keys.state_value],
    )

    log_probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float).log()

    # batched_probs, action_indices should not be relevant for the loss
    def embedded_forward_mock(
        current_embedding: ObjectEmbedding,
        successor_embedding: ObjectEmbedding,
        num_successors: torch.Tensor,
    ):
        assert (num_successors == real_num_successors).all()
        assert current_embedding == current_object_embeddings
        assert successor_embedding == successor_object_embeddings
        batched_probs = [
            torch.rand(
                (num_successors[i].item(),), dtype=torch.float, requires_grad=True
            ).softmax(dim=0)
            for i in range(BATCH_SIZE)
        ]
        action_indices = torch.zeros((BATCH_SIZE,), dtype=torch.long)
        return batched_probs, action_indices, log_probs

    # return batched_probs, action_indices and log_probs
    actor_critic_mock = mockito.mock(
        {
            "embedded_forward": embedded_forward_mock,
            "keys": ActorCritic.default_keys,
        },
        spec=ActorCritic,
    )

    # Setup Loss and PolicyGradientModule
    loss = ActorCriticLoss(operator_mock, reduction="mean", loss_critic_type="l2")
    loss.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)
    optimizer_mock = mockito.mock(spec=torch.optim.Optimizer)

    adapter = PolicyGradientLitModule(
        gnn_mock, actor_critic_mock, loss=loss, optim=optimizer_mock
    )

    # Execute main test component
    loss = adapter.training_step(batched_tuple)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    # critic_loss
    # rewards + gamma * next_values
    targets = torch.full((BATCH_SIZE,), fill_value=-1 + GAMMA * 1)
    targets[0] = 0.0  # first is terminal
    # We use loss_critic_type=l2 and reduction = mean -> mse_loss
    critic_loss = torch.nn.functional.mse_loss(expected_current_values, targets)
    advantage = targets - expected_current_values

    # actor_loss
    actor_loss = (-log_probs * advantage).mean()

    assert torch.allclose(loss, critic_loss + actor_loss)
