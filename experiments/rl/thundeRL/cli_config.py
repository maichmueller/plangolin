import functools
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from lightning import Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    OptimizerCallable,
    SaveConfigCallback,
)
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators

from experiments.rl.configs.trainer import optimal_values
from experiments.rl.data_layout import InputData, OutputData
from rgnet import HeteroGNN, HeteroGraphEncoder
from rgnet.rl import ActorCritic, ActorCriticLoss
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter
from rgnet.rl.thundeRL.validation import (
    CriticValidation,
    PolicyValidation,
    optimal_policy,
)


class resolve_optim:

    def __init__(
        self,
        agent: ActorCritic,
        optimizer: OptimizerCallable,  # already partly initialized from cli
        lr_actor: Optional[float] = None,
        lr_critic: Optional[float] = None,
        lr_embedding: Optional[float] = None,
    ):
        lr_parameters: List[Dict] = []  # parameter specific learning rate
        plain_parameter: List = []  # no specific learning rate
        for specific_lr, params in [
            (lr_actor, agent.actor_net.parameters()),
            (lr_critic, agent.value_operator.module.parameters()),
            (
                lr_embedding,
                (
                    agent.embedding_module.gnn.parameters()
                    if agent.embedding_module
                    else []
                ),
            ),
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


def configure_loss(loss: ActorCriticLoss, estimator: ValueEstimatorConfig):
    loss.make_value_estimator(value_type=estimator.estimator, gamma=estimator.gamma)
    return loss


@functools.cache
def configure_eval_callbacks(
    input_data: InputData, gamma: float, value_operator: ValueOperator
):
    if not input_data.validation_problems:
        return []
    # Create the hooks
    optimal_values_dict: Dict[int, torch.Tensor] = {
        i: optimal_values(space, gamma)
        for i, space in enumerate(input_data.validation_spaces)
    }
    critic_validation = CriticValidation(
        optimal_values_dict=optimal_values_dict, value_operator=value_operator
    )
    policy_validation = PolicyValidation(
        optimal={
            i: optimal_policy(space)
            for i, space in enumerate(input_data.validation_spaces)
        }
    )
    return [critic_validation, policy_validation]


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
        parser.add_class_arguments(resolve_optim, "optimizer", as_positional=True)
        parser.add_class_arguments(
            ValueEstimatorConfig, "value_estimator", as_positional=True
        )

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

        parser.link_arguments("agent", "optimizer.agent", apply_on="instantiate")
        parser.link_arguments(
            "optimizer.optimizer", "model.optim", apply_on="instantiate"
        )
        # Validation hook links
        parser.link_arguments(
            source=(
                "data_layout.input_data",
                "value_estimator.gamma",
                "agent.value_operator",
            ),
            target="model.validation_hooks",
            compute_fn=configure_eval_callbacks,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source=(
                "data_layout.input_data",
                "value_estimator.gamma",
                "agent.value_operator",
            ),
            target="trainer.callbacks",
            compute_fn=configure_eval_callbacks,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.default_root_dir",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )

    def before_fit(self):
        self.trainer.logger.log_hyperparams(self.config)
