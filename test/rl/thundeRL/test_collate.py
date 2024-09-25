import torch.utils.data
from fixtures import fresh_drive, medium_blocks

from rgnet.rl.thundeRL.collate import collate_fn


def test(tmp_path, medium_blocks, fresh_drive):
    space = medium_blocks[0]

    loader = torch.utils.data.DataLoader(
        fresh_drive, collate_fn=collate_fn, batch_size=25
    )
    batches = [batch for batch in loader]
