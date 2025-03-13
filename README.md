# RGNet

An open source library for Machine Learning on Planning problems.
Core concepts include

- **Reinforcement Learning**
- **Graph Neural Networks**
- **Generalized Planning**

## Installation

The project can be installed via pip except pymimir. We are currently in the migration
process but currently rely on
the [archive/v1](https://github.com/simon-stahlberg/mimir/tree/archive/v1) branch of
pymimir.
On the cluster you will most likely need to clone the mimir-repository and
run `pip install .` in the root directory of the cloned repo.
Additional dependencies for testing and development can be installed via `pip install
-e .[dev]`.

## Usage

Currently three different use cases are implemented with varying degrees of support.

#### 1. Supervised Learning

#### 2. Reinforcement Learning with arbitrary rollout-length and arbitrarily big problems.

#### 3. An optimized RL version with one-step rollouts and a fully enumerated state space.

- See `docs/thundeRL/README.md` for further details

## Roadmap

#TODO

## Techstack

- [TorchRL](https://github.com/pytorch/rl): as modular open source RL framework
  - Support for `NonTensorData`: The observation space consists of complex objects.
  - Actions vary by state and the total number is to big for masking.
  - Aligns with the pytorch ecosystem
- [PyTorch](https://pytorch.org/): as the deep learning framework.
- [PyTorch-Geometric](https://github.com/pyg-team/pytorch_geometric): for graph neural
  networks.
- [PyTorch Lightning](https://www.pytorchlightning.ai/): as the training framework.
  - [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) / [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/index.html#jsonargparse.ArgumentParser.add_instantiator)
    as cli managing tool. Allowing clean experiment configuration with yaml files.
- [Pymimir](https://github.com/simon-stahlberg/mimir): as underlying planning tool.
  - Parsing PDDL files.
  - Enumerating the grounded state space of a planning problem.
- [Wandb](https://wandb.ai/site): for experiment tracking.

## Contributing

#TODO

## Authors and acknowledgment

#TODO
