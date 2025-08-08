from torch.nn import Module


class ResidualModule(Module):
    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
