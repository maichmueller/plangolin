import itertools
from test.fixtures import *  # noqa: F401, F403

import torch.utils.data

from rgnet.rl.thundeRL.collate import to_transitions_batch


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
    assert all(isinstance(batch, tuple) and len(batch) == 3 for batch in batches)

    flattened_indices = itertools.chain.from_iterable(
        [batch[0].idx.tolist() for batch in batches]
    )
    assert set(flattened_indices) == set(range(len(space)))
