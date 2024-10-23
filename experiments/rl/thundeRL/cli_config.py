import dataclasses
import warnings
from argparse import Namespace
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import torch
from lightning import Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    OptimizerCallable,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import WandbLogger
from torchrl.objectives import ValueEstimators

from experiments.rl.configs.trainer import optimal_values
from experiments.rl.data_layout import InputData, OutputData
from rgnet import HeteroGNN, HeteroGraphEncoder
from rgnet.rl import ActorCritic, ActorCriticLoss
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter
from rgnet.rl.thundeRL.validation import CriticValidation, optimal_policy  # noqa: F401


class OptimizerSetup:

    def __init__(
        self,
        agent: ActorCritic,
        optimizer: OptimizerCallable,  # already partly initialized from cli
        gnn: Optional[HeteroGNN] = None,
        lr_actor: Optional[float] = None,
        lr_critic: Optional[float] = None,
        lr_embedding: Optional[float] = None,
    ):
        lr_parameters: List[Dict] = []  # parameter specific learning rate
        plain_parameter: List = []  # no specific learning rate
        if agent.embedding_module is None and gnn is None:
            warnings.warn("No parameter for the embedding found.")
            gnn_params = []
        elif agent.embedding_module is not None:
            gnn_params = agent.embedding_module.gnn.parameters()
        else:
            gnn_params = gnn.parameters()
        for specific_lr, params in [
            (lr_actor, agent.actor_net.parameters()),
            (lr_critic, agent.value_operator.parameters()),
            (lr_embedding, gnn_params),
        ]:
            if specific_lr and params:
                lr_parameters.append(
                    {
                        "params": params,
                        "lr": specific_lr,
                    }
                )
            elif params:
                plain_parameter.extend(params)
        if plain_parameter:
            lr_parameters.append({"params": plain_parameter})
        self.optimizer = optimizer(lr_parameters)


class ValueEstimatorConfig:

    def __init__(
        self, gamma: float, estimator_type: ValueEstimators = ValueEstimators.TD0
    ):
        self.gamma = gamma
        self.estimator = estimator_type


@dataclasses.dataclass
class WandbExtraParameter:
    watch_model: Optional[bool] = True  # whether to watch the model gradients
    log_frequency: int = 100  # the frequency for watch


def configure_loss(loss: ActorCriticLoss, estimator: ValueEstimatorConfig):
    loss.make_value_estimator(value_type=estimator.estimator, gamma=estimator.gamma)
    return loss


def optimal_policy_dict(input_data: InputData):
    return {
        i: optimal_policy(space) for i, space in enumerate(input_data.validation_spaces)
    }


def optimal_values_dict(input_data: InputData, gamma: float) -> Dict[int, torch.Tensor]:
    return {
        i: optimal_values(space, gamma)
        for i, space in enumerate(input_data.validation_spaces)
    }


def validation_dataloader_names(input_data: InputData) -> Optional[Dict[int, str]]:
    if input_data.validation_problems is None:
        return None

    return {i: p.name for i, p in enumerate(input_data.validation_problems)}


class ThundeRLCLI(LightningCLI):

    def __init__(
        self,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[
            Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
    ) -> None:
        super().__init__(
            LightningAdapter,
            ThundeRLDataModule,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            args,
            run,
            auto_configure_optimizers=False,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser = value_estimator_module.add_parser_args(parser)
        parser.add_class_arguments(HeteroGNN, "hetero_gnn", as_positional=True)
        parser.add_class_arguments(
            HeteroGraphEncoder, "encoding", as_positional=True, skip={"node_factory"}
        )
        parser.add_argument(
            "--data_layout.root_dir", type=Optional[PathLike], default=None
        )
        parser.add_argument(
            "--test_max_steps",
            type=int,
            default=100,
        )
        parser.add_class_arguments(
            InputData, "data_layout.input_data", as_positional=True
        )
        parser.add_dataclass_arguments(OutputData, "data_layout.output_data")
        parser.add_class_arguments(
            ActorCriticLoss,
            "ac_loss",
            as_positional=True,
            skip={"clone_tensordict", "keys"},
        )
        parser.add_class_arguments(
            ActorCritic, "agent", as_positional=True, skip={"keys"}
        )
        parser.add_class_arguments(
            OptimizerSetup, "optimizer_setup", as_positional=True
        )
        parser.add_class_arguments(
            ValueEstimatorConfig, "value_estimator", as_positional=True
        )
        parser.add_dataclass_arguments(WandbExtraParameter, "wandb_extra")
        # Link arguments
        parser.link_arguments(
            "data_layout.root_dir", "data_layout.input_data.root_dir", apply_on="parse"
        )
        parser.link_arguments(
            "data_layout.root_dir", "data_layout.output_data.root_dir", apply_on="parse"
        )
        parser.link_arguments(
            "data_layout.input_data.domain_name",
            "data_layout.output_data.domain_name",
            apply_on="parse",
        )
        parser.link_arguments(
            "data_layout.input_data.domain", "encoding.domain", apply_on="instantiate"
        )
        parser.link_arguments(
            "encoding.obj_type_id", "hetero_gnn.obj_type_id", apply_on="instantiate"
        )
        parser.link_arguments(
            "encoding.arity_dict", "hetero_gnn.arity_dict", apply_on="instantiate"
        )
        parser.link_arguments(
            "data_layout.input_data", "data.input_data", apply_on="instantiate"
        )
        parser.link_arguments(
            "hetero_gnn.hidden_size", "agent.hidden_size", apply_on="parse"
        )
        parser.link_arguments("value_estimator.gamma", "data.gamma", apply_on="parse")
        parser.link_arguments("hetero_gnn", "model.gnn", apply_on="instantiate")
        parser.link_arguments(
            "agent.value_operator", "ac_loss.critic_network", apply_on="instantiate"
        )
        # Model links
        parser.link_arguments("agent", "model.actor_critic", apply_on="instantiate")

        parser.link_arguments(
            ("ac_loss", "value_estimator"),
            "model.loss",
            apply_on="instantiate",
            compute_fn=configure_loss,
        )

        parser.link_arguments("agent", "optimizer_setup.agent", apply_on="instantiate")
        parser.link_arguments(
            "hetero_gnn", "optimizer_setup.gnn", apply_on="instantiate"
        )
        parser.link_arguments(
            "optimizer_setup.optimizer", "model.optim", apply_on="instantiate"
        )

        # Trainer / logger links
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.output_data.experiment_name",
            target="trainer.logger.init_args.name",
            apply_on="instantiate",
        )

        # Validation callback links
        # NOTE it seems like you can't have two callbacks which have different parameter
        # of the same name.
        # Not a problem for dataloader_names as it is the same for all of them.
        parser.link_arguments(
            source="agent.value_operator",
            target="model.validation_hooks.init_args.value_operator",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source=("data_layout.input_data", "value_estimator.gamma"),
            target="model.validation_hooks.init_args.optimal_values_dict",
            compute_fn=optimal_values_dict,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.input_data",
            target="model.validation_hooks.init_args.optimal_policy_dict",
            compute_fn=optimal_policy_dict,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="model.validation_hooks.init_args.save_dir",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.input_data",
            target="model.validation_hooks.init_args.dataloader_names",
            compute_fn=validation_dataloader_names,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.default_root_dir",
            apply_on="instantiate",
        )

    def instantiate_trainer(self, **kwargs: Dict) -> Trainer:
        """
        We need to add the validation callbacks of the model to the trainer.
        The problem is that we have a list of callbacks, and we can't extend
        the list of callbacks provided via the config using jsonargparse.
        LightningCLI offers an extra way via "forced callbacks" but that doesn't work with lists too.
        Therefore, we manually add our model callbacks to the extra callbacks.
        """
        if isinstance(self.model, LightningAdapter) and self.model.validation_hooks:
            model_callbacks = self.model.validation_hooks
            extra_callbacks = [
                self._get(self.config_init, c)
                for c in self._parser(self.subcommand).callback_keys
            ]
            extra_callbacks.extend(model_callbacks)
            trainer_config = {
                **self._get(self.config_init, "trainer", default={}),
                **kwargs,
            }
            return self._instantiate_trainer(trainer_config, extra_callbacks)

        return super().instantiate_trainer(**kwargs)

    def convert_to_nested_dict(self, config: Namespace):
        """Lightning converts nested namespaces to strings"""
        mapping: Dict = vars(config).copy()
        for key, item in mapping.items():
            if isinstance(item, Namespace):
                mapping[key] = self.convert_to_nested_dict(item)
            if isinstance(item, Sequence) and not isinstance(item, str):
                mapping[key] = [
                    (
                        sequence_item
                        if not isinstance(sequence_item, Namespace)
                        else self.convert_to_nested_dict(sequence_item)
                    )
                    for sequence_item in item
                ]

        return mapping

    def before_fit(self):
        self.trainer.logger.log_hyperparams(
            self.convert_to_nested_dict(self.config["fit"])
        )
        wandb_extra: WandbExtraParameter = self.config_init["fit"]["wandb_extra"]
        if wandb_extra.watch_model and isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger.watch(self.model, log_freq=wandb_extra.log_frequency)
