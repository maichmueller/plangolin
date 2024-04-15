from typing import Optional

import torch
import torch.nn
import torch.nn.functional as F
import torch_geometric as pyg

from rgnet.model import PureGNN
from rgnet.utils import get_device_cuda_if_possible


def training(
    dataset: pyg.data.Dataset, device: Optional[torch.DeviceObjType] = None
) -> PureGNN:
    device = device if (device is not None) else get_device_cuda_if_possible()
    seventy_percent = int(0.7 * len(dataset))
    test_set, train_set = dataset[:seventy_percent], dataset[seventy_percent:]
    loader = pyg.loader.DataLoader(test_set, 32, shuffle=True)

    model: PureGNN = PureGNN(in_channel=1, embedding_size=32, num_layer=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch, data in enumerate(loader):
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.l1_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
    return model


def evaluate(
    model: PureGNN,
    dataset: pyg.data.Dataset,
    device: Optional[torch.DeviceObjType] = None,
):
    device = device if (device is not None) else get_device_cuda_if_possible()
    test_loader = pyg.loader.DataLoader(dataset, 32, shuffle=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            eval_loss = F.l1_loss(out.squeeze(), data.y)
            print(f"Test l1-loss value : {eval_loss.item():.4f}")
