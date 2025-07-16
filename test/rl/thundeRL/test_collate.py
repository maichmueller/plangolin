import itertools
from test.fixtures import *  # noqa: F401, F403

import torch.utils.data

from rgnet.rl.envs import SuccessorEnvironment
from rgnet.rl.thundeRL.collate import (
    StatefulCollater,
    to_iw_transitions_batch,
    to_transitions_batch,
)
from rgnet.utils.batching import batched
from xmimir import XSuccessorGenerator, XTransition, iw
from xmimir.iw import RandomizedExpansion


def test(tmp_path, medium_blocks, fresh_flashdrive_medium_blocks):
    space = medium_blocks[0]
    assert len(space) == 125

    loader = torch.utils.data.DataLoader(
        fresh_flashdrive_medium_blocks,
        collate_fn=to_transitions_batch,
        batch_size=25,
    )
    batches = [batch for batch in loader]
    assert len(batches) == 5
    assert all(isinstance(batch, tuple) and len(batch) == 4 for batch in batches)

    flattened_indices = itertools.chain.from_iterable(
        [batch[0].idx.tolist() for batch in batches]
    )
    assert set(flattened_indices) == set(range(len(space)))


def test_iw_transitions_batch(tmp_path, medium_blocks):
    space = medium_blocks[0]
    assert len(space) == 125

    # torch.multiprocessing.set_sharing_strategy("file_descriptor")

    iw_search = iw.IWSearch(
        width=1,
        expansion_strategy=RandomizedExpansion(seed=0),
    )
    drive = make_fresh_flashdrive(
        tmp_path,
        domain="blocks",
        problem="medium.pddl",
        force_reload=False,
        attribute_getters={
            "action_history": "rgnet.rl.data.flash_drive.attr_getters.action_history_datapack",
            "domain_path": "rgnet.rl.data.flash_drive.attr_getters.domain_path",
            "problem_path": "rgnet.rl.data.flash_drive.attr_getters.problem_path",
        },
    )

    loader = torch.utils.data.DataLoader(
        drive,
        collate_fn=StatefulCollater(
            to_iw_transitions_batch,
            iw_search=iw_search,
            reward_function=drive.reward_function,
            encoder_factory=EncoderFactory(HeteroGraphEncoder),
        ),
        batch_size=25,
        # num_workers=2,
        shuffle=False,
    )
    batches = [batch for batch in loader]
    assert len(batches) == 5
    assert all(isinstance(batch, tuple) and len(batch) == 4 for batch in batches)

    flattened_indices = itertools.chain.from_iterable(
        [batch[0].idx.tolist() for batch in batches]
    )
    assert set(flattened_indices) == set(range(len(space)))

    iw_search = iw.IWSearch(
        width=1,
        expansion_strategy=RandomizedExpansion(seed=0),
    )
    space, domain, problem = medium_blocks
    new_domain, new_problem = xmi.parse(domain.filepath, problem.filepath)
    succ_gen = XSuccessorGenerator(new_problem)
    succ_env = SuccessorEnvironment(
        generators=[succ_gen],
        reward_function=drive.reward_function,
        batch_size=1,
        reset=True,
    )
    encoder = HeteroGraphEncoder(domain)
    data_list = []
    for state_d in drive:
        state = state_d.action_history.reconstruct(succ_gen)
        state_data = encoder.to_pyg_data(encoder.encode(state))
        data_list.append(state_data)
        collector = iw.CollectorHook()
        iw_search.solve(
            start_state=state,
            successor_generator=succ_gen,
            stop_on_goal=False,
            novel_hook=collector,
        )
        transitions, targets = (
            [],
            [],
        )
        for node in collector.nodes:
            successor = node.state
            trace = node.trace
            iw_transition = XTransition.make_hollow(
                state, [t.action for t in trace], successor
            )
            transitions.append(iw_transition)
            successor_data = encoder.to_pyg_data(encoder.encode(successor))
            targets.append(successor_data)
        reward, done = succ_env.get_reward_and_done(transitions=transitions)
        state_data.reward = reward
        state_data.done = done
        state_data.targets = targets
    for data_l, test_batch in zip(batched(data_list, 25), batches):
        batch, succ_batch, num_successors, info = to_transitions_batch(
            list(data_l), exclude_keys=["targets"]
        )
        test_current_batch, test_succ_batch, test_num_successors, test_info = test_batch
        assert torch.isclose(
            torch.mean(batch.reward), torch.mean(test_current_batch.reward), atol=1.0
        )
        assert torch.isclose(
            torch.std(batch.reward), torch.std(test_current_batch.reward), atol=0.1
        )
        assert torch.isclose(
            torch.mean(batch.done.float()),
            torch.mean(test_current_batch.done.float()),
            atol=0.1,
        )
        assert torch.isclose(
            torch.std(batch.done.float()),
            torch.std(test_current_batch.done.float()),
            atol=0.1,
        )
        assert torch.isclose(
            torch.mean(num_successors.float()),
            torch.mean(test_num_successors.float()),
            atol=1.0,
        )
