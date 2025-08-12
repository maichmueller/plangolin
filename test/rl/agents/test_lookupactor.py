import operator
from test.fixtures import medium_blocks  # noqa: F401

from plangolin.algorithms import optimal_policy
from plangolin.rl.agents import LookupPolicyActor
from plangolin.rl.envs import PlanningEnvironment


def test_lookup_policy_actor_deterministic():
    import torch
    from tensordict import TensorDict

    torch.manual_seed(0)

    # Two states with different action cardinalities
    # state 0 -> always pick action 0
    # state 1 -> always pick action 1 (of length 3 vector)
    probs_list = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    ]

    actor = LookupPolicyActor(
        probs_list=probs_list,
        problem=None,  # not used by actor logic
        env_keys=PlanningEnvironment.default_keys,
        idx_of_state="state",
    )

    td = TensorDict(
        {"state": torch.tensor([0, 1, 1, 0], dtype=torch.long)}, batch_size=[4]
    )
    out = actor(td)

    actions = out["action"].tolist()
    assert actions == [0, 1, 1, 0]
    assert out["action"].dtype == torch.long


def test_lookup_policy_actor_padding_mask_never_sample_padded():
    import torch
    from tensordict import TensorDict

    torch.manual_seed(123)

    # state 0 has 2 valid actions, state 1 has 1 valid action
    probs_list = [
        torch.tensor([0.2, 0.8], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    ]

    actor = LookupPolicyActor(
        probs_list=probs_list,
        problem=None,  # shouldn't trigger an error (but may change)
        env_keys=PlanningEnvironment.default_keys,
        idx_of_state="my_custom_key",
    )

    # Build a batch with many repetitions to increase chances of catching padding bugs
    batch0 = torch.zeros(1000, dtype=torch.long)
    batch1 = torch.ones(1000, dtype=torch.long)
    states = torch.cat([batch0, batch1], dim=0)

    td = TensorDict({"my_custom_key": states}, batch_size=[states.shape[0]])
    out = actor(td)

    actions = out["action"]
    # For state 0, valid actions are {0,1}
    a0 = actions[:1000]
    assert a0.min().item() >= 0 and a0.max().item() <= 1
    # For state 1, only action 0 is valid
    a1 = actions[1000:]
    assert torch.all(a1 == 0)


def test_lookup_actor_astar_medium_deterministic(medium_blocks):
    import torch
    from tensordict import TensorDict

    space, _domain, _problem = medium_blocks
    optimal_successors = optimal_policy(space)
    policy = [
        torch.zeros(space.forward_transition_count(space[i]))
        for i, probs in sorted(optimal_successors.items(), key=operator.itemgetter(0))
    ]

    for i, probs in enumerate(policy):
        for opt_idx in optimal_successors[i]:
            probs[opt_idx] = 1.0

    actor = LookupPolicyActor(
        probs_list=policy,
        problem=_problem,
        env_keys=PlanningEnvironment.default_keys,
        idx_of_state=space,
    )

    td = TensorDict(
        {PlanningEnvironment.default_keys.state: list(space)}, batch_size=[len(space)]
    )
    out = actor(td)
    actions = out[PlanningEnvironment.default_keys.action]

    # Deterministic: always the optimal action, never padded
    for i, s in enumerate(space):
        a = int(actions[i])
        best_action_inds = (policy[s.index] == policy[s.index].max()).nonzero()
        assert a in best_action_inds


def test_lookup_actor_astar_medium_stochastic(medium_blocks):
    import torch
    from tensordict import TensorDict

    space, _domain, _problem = medium_blocks
    eps = 0.2
    optimal_successors = optimal_policy(space)
    policy = [
        torch.zeros(space.forward_transition_count(space[i]))
        for i, probs in sorted(optimal_successors.items(), key=operator.itemgetter(0))
    ]
    for i, probs in enumerate(policy):
        for opt_idx in optimal_successors[i]:
            probs[opt_idx] = 1.0
        policy[i] = probs * (1 - eps) + eps / len(probs)

    actor = LookupPolicyActor(
        probs_list=policy,
        problem=_problem,
        env_keys=PlanningEnvironment.default_keys,
        idx_of_state="state_idx",
    )

    idx_batch = torch.tensor([s.index for s in space], dtype=torch.long)
    td = TensorDict({"state_idx": idx_batch}, batch_size=[idx_batch.shape[0]])

    # Sample multiple times and accumulate frequencies
    R = 128
    correct = torch.zeros(len(space), dtype=torch.int32)
    max_seen = torch.zeros(len(space), dtype=torch.int64)
    for _ in range(R):
        out = actor(td.clone())
        a = out["action"].to(torch.int64)
        # update max seen for padded-guard
        max_seen = torch.maximum(max_seen, a)
        # count correct
        for i, s in enumerate(space):
            best_action_inds = (policy[s.index] == policy[s.index].max()).nonzero()
            if int(a[i]) in best_action_inds:
                correct[i] += 1

    # Optimal should dominate; threshold conservative for small K
    frac = correct.float() / R
    for i, s in enumerate(space):
        K = space.forward_transition_count(s)
        # skip degenerate K==1
        if K == 1:
            continue
        # Expected p* = (1-eps) + eps/K; assert at least 0.6 margin
        assert frac[i] >= 0.6
        # Never sample padded actions
        assert int(max_seen[i]) < K
