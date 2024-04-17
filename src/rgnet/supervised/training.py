import logging
import pathlib
import time
from typing import Optional, Tuple

import torch
import torch.nn
import torch.nn.functional as F
import torch_geometric as pyg

from rgnet.model import PureGNN
from rgnet.utils import get_device_cuda_if_possible, path_of_str, time_delta_now


class Trainer:

    def __init__(
        self,
        train_set: pyg.data.Dataset,
        test_set: pyg.data.Dataset,
        epochs: int = 1,
        batch_size: int = 64,
        embedding_size: int = 32,
        num_layer: int = 24,
        evaluate_after_epoch: bool = True,
        save_file: str | pathlib.Path | None = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = PureGNN(
            in_channel=1, embedding_size=embedding_size, num_layer=num_layer
        )
        self.evaluate_after_epoch = evaluate_after_epoch
        self.save_file = None if save_file is None else path_of_str(save_file)
        self.device = device if (device is not None) else get_device_cuda_if_possible()

    def train(self):
        loader = pyg.loader.DataLoader(self.train_set, self.batch_size, shuffle=True)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            for i, data in enumerate(loader):
                data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.batch)
                loss = F.l1_loss(out.squeeze(), data.y)
                loss.backward()
                optimizer.step()

            logging.info(
                f"Completed {epoch+1} of {self.epochs} epochs in {time_delta_now(epoch_start_time)}"
            )

            if self.evaluate_after_epoch:
                mae, num_states = self.evaluate()
                logging.info(
                    f"After {epoch+1} epochs got average mae of {mae / num_states} over {num_states} states"
                )
            if self.save_file is not None:
                torch.save(self.model, self.save_file)

    def evaluate(self) -> Tuple[float, int]:
        """
        Evaluate the current model over the testing-set.
        :return: The summed l1-loss over all examples, the number of examples
        """
        test_loader = pyg.loader.DataLoader(
            self.test_set, self.batch_size, shuffle=False
        )
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            mae = 0.0
            num_states = 0
            # Iterate in batches over the training/test dataset.
            for data in test_loader:
                data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                eval_loss = F.l1_loss(out.squeeze(), data.y)
                mae += eval_loss.item()
                num_states += len(data.y)
            return mae, num_states
