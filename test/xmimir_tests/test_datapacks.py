from collections import defaultdict
from test.fixtures import medium_blocks  # noqa: F401, F403

import numpy as np

from rgnet.logging_setup import get_logger
from xmimir import (
    ActionDataPack,
    ActionHistoryDataPack,
    XActionGenerator,
    XSuccessorGenerator,
    parse,
)


def test_actionpack(medium_blocks):
    space, domain, problem = medium_blocks
    transitions = sum([list(space.forward_transitions(state)) for state in space], [])
    action_packs = [ActionDataPack(t.action) for t in transitions]

    new_domain, new_prob = parse(domain.filepath, problem.filepath)
    new_action_gen = XActionGenerator(new_prob)
    reconstruced_actions = [
        action_pack.reconstruct(new_action_gen) for action_pack in action_packs
    ]
    assert len(reconstruced_actions) == len(action_packs)
    logger = get_logger(__name__)
    for recon_action, transition in zip(reconstruced_actions, transitions):
        orig_action = transition.action
        assert recon_action.action_schema.name == orig_action.action_schema.name
        assert [o.get_name() for o in recon_action.objects] == [
            o.get_name() for o in orig_action.objects
        ]
        if recon_action.index != transition.action.index:
            logger.info(
                "(Expected) Action index mismatch: %s != %s",
                recon_action.index,
                transition.action.index,
            )


def test_actionhistorypack(medium_blocks):
    space, domain, problem = medium_blocks

    states_to_sample = 100
    rng = np.random.default_rng(42)
    action_history = defaultdict(list)
    states_sampled = []
    while len(states_sampled) < states_to_sample:
        state = space.initial_state
        history = []
        for step in range(rng.integers(1, 40)):
            if len(states_sampled) >= states_to_sample:
                break
            transitions = list(space.forward_transitions(state))
            transition = transitions[rng.integers(0, len(transitions))]
            history.append(transition.action)
            state = transition.target
            if state not in states_sampled:
                states_sampled.append(state)
                action_history[state] = history.copy()

    new_domain, new_prob = parse(domain.filepath, problem.filepath)
    new_state_gen = XSuccessorGenerator(new_prob)
    reconstructed_states = [
        ActionHistoryDataPack(action_history[state]).reconstruct(new_state_gen)
        for state in states_sampled
    ]

    assert len(reconstructed_states) == len(states_sampled)
    logger = get_logger(__name__)
    for recon_state, orig_state in zip(reconstructed_states, states_sampled):
        assert recon_state.semantic_eq(orig_state)
        if recon_state.index != orig_state.index:
            logger.info(
                "(Expected) Action index mismatch: %s != %s",
                recon_state.index,
                orig_state.index,
            )
