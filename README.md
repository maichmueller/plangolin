<p align="center">
  <img src="media/plangolin_logo.png" alt="plangolin_with_title" width="300px">
</p>

[![pytest](https://github.com/maichmueller/plangolin/actions/workflows/ci.yml/badge.svg)](https://github.com/maichmueller/plangolin/actions/workflows/ci.yml)

______________________________________________________________________
An open source research library for Machine/Reinforcement Learning on Classical Planning problems using Graph Neural Networks.

This library is purely research-focused. APIs and interfaces may change.

## Installation

The project can be installed via pip `pip install .`

For a development installation run `pip install -e '.[dev]'`

## Usage

Currently two different use cases are implemented with different degrees of support.

#### 1. Planning problems as deterministic MDPs with behaviour-based exploration.

This direction is the usual RL design pattern, where the agent generates data by interacting with the environment.
A common downside of this approach is that exploration is costly in time and resources, rendering this approach slow.

The implementation is found in the `rgnet.rl` module and is based on the `torchrl` framework.

#### 2. Planning problems as deterministic MDP datasets on demand.

This direction is a more efficient approach, where the agent is trained on a pre-computed dataset of planning problems.
Here, the agent learns by loading environment data on demand from a dataset,
essentially turning the RL problem into a hybrid supervised-RL learning problem:
RL update rules, but supervised data loading.

The implementation is found in the `rgnet.rl.thundeRL` module of this library,
utilizing `pytorch-lightning` for vastly faster training and evaluation.
See `docs/thundeRL/README.md` for further details on this.

See the `examples` directory for example usage of the library for both directions.

## Techstack

- [TorchRL](https://github.com/pytorch/rl): as modular open source RL framework
  - Support for `NonTensorData`: The observation space consists of complex objects.
  - Irregular state-dependant action-space without apriori known upper limit (no boxes, no masking techniques applicable).
  - Aligns with the pytorch ecosystem
- [PyTorch-Geometric](https://github.com/pyg-team/pytorch_geometric): graph neural networks basis.
- [PyTorch Lightning](https://www.pytorchlightning.ai/): as the training framework.
  - [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) / [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/index.html#jsonargparse.ArgumentParser.add_instantiator)
    as cli managing tool. Allowing clean experiment configuration with yaml files.
- [Pymimir](https://github.com/simon-stahlberg/mimir): as underlying planning tool (parsing PDDLs, grounding/lifting STRIPS-states, etc.).

## Contributing

Ideas and suggestions are welcome! Please open an issue or a merge request.
