import dataclasses
import warnings
from argparse import Namespace
from functools import cache
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union

import torch
from jsonargparse import lazy_instance
from lightning import Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    OptimizerCallable,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import WandbLogger
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import ValueEstimators

# avoids specifying full class_path for encoder in cli
from rgnet.encoding import (  # noqa: F401
    ColorGraphEncoder,
    DirectGraphEncoder,
    GraphEncoderBase,
    HeteroGraphEncoder,
)
from rgnet.encoding.base_encoder import EncoderFactory

# avoids specifying full class_path for model.gnn in cli
from rgnet.models import HeteroGNN, VanillaGNN  # noqa: F401
from rgnet.rl.agents import ActorCritic
from rgnet.rl.data_layout import InputData, OutputData
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.losses import (  # noqa: F401
    ActorCriticLoss,
    AllActionsLoss,
    AllActionsValueEstimator,
    CriticLoss,
)
from rgnet.rl.losses.all_actions_estimator import KeyBasedProvider
from rgnet.rl.optimality_utils import bellman_optimal_values, optimal_policy
from rgnet.rl.reward import RewardFunction, UnitReward
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule
from rgnet.rl.thundeRL.policy_gradient_lit_module import PolicyGradientLitModule

# Import before the cli makes it possible to specify only the class and not the
# full class path for model.validation_hooks in the cli config.
from rgnet.rl.thundeRL.validation import (  # noqa: F401
    CriticValidation,
    PolicyEntropy,
    PolicyValidation,
    ProbsCollector,
    ProbsStoreCallback,
)


class OptimizerSetup:
    def __init__(
        self,
        agent: ActorCritic,
        optimizer: OptimizerCallable,  # already partly initialized from cli
        gnn: Optional[torch.nn.Module] = None,
        lr_actor: Optional[float] = None,
        lr_critic: Optional[float] = None,
        lr_embedding: Optional[float] = None,
    ):
        lr_parameters: List[Dict] = []  # parameter-specific learning rate
        plain_parameter: List = []  # no specific learning rate
        if agent.embedding_module is None and gnn is None:
            warnings.warn("No parameter for the embedding found.")
            gnn_params = []
        elif agent.embedding_module is not None:
            gnn_params = agent.embedding_module.gnn.parameters()
        else:
            gnn_params = gnn.parameters()
        for specific_lr, params in [
            (
                lr_actor,
                chain(
                    agent.actor_net_probs.parameters(),
                    agent.actor_objects_net.parameters(),
                ),
            ),
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


@dataclasses.dataclass
class WandbExtraParameter:
    watch_model: Optional[bool] = True  # whether to watch the model gradients
    log_frequency: int = 100  # the frequency for watch
    log_code: bool = False  # whether to save the code as wandb artifact


def optimal_policy_dict(input_data: InputData):
    return {
        i: optimal_policy(space) for i, space in enumerate(input_data.validation_spaces)
    }


@cache
def discounted_optimal_values_dict(
    input_data: InputData, reward_func: RewardFunction
) -> Dict[int, torch.Tensor]:
    return {
        i: bellman_optimal_values(
            ExpandedStateSpaceEnv(space, reward_function=reward_func, reset=True)
        )
        for i, space in enumerate(input_data.validation_spaces)
    }


def validation_dataloader_names(input_data: InputData) -> Optional[Dict[int, str]]:
    if input_data.validation_problems is None:
        return None

    return {i: p.name for i, p in enumerate(input_data.validation_problems)}


class ValueEstimatorConfig:
    def __init__(
        self,
        gamma: float,
        estimator_type: (
            ValueEstimators | Literal["AllActionsValueEstimator"]
        ) = ValueEstimators.TD0,
    ):
        self.gamma = gamma
        self.estimator_type = estimator_type


def configure_loss(loss: CriticLoss, estimator_config: ValueEstimatorConfig):
    """
    We need to configure the estimator after the loss was instantiated via make_value_estimator.
    This is why we have the estimator and loss as separate keys and link them together for
    configuring `model.loss`.
    """
    hyperparameter = dict()
    if estimator_config.estimator_type == "AllActionsValueEstimator":
        hyperparameter["reward_done_provider"] = KeyBasedProvider(
            reward_key=PolicyGradientLitModule.default_keys.all_rewards,
            done_key=PolicyGradientLitModule.default_keys.all_dones,
        )
    hyperparameter["shifted"] = True
    hyperparameter["gamma"] = estimator_config.gamma

    loss.make_value_estimator(estimator_config.estimator_type, **hyperparameter)
    return loss


@dataclasses.dataclass
class TestSetup:
    """
    Define additional parameter used for testing the agent.
    Args:
        max_steps: The maximum number of steps the agent is allowed to take to solve any problem.
            (default: 100)
        exploration_type: How the actions should be sampled. RANDOM means the probability
            distribution over successor states is sampled and MODE will take the arg-max.
            (default: ExplorationType.MODE)
    """

    max_steps: int = 100
    exploration_type: ExplorationType = ExplorationType.MODE


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
            PolicyGradientLitModule,
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
        parser.add_subclass_arguments(GraphEncoderBase, "encoder")

        parser.add_class_arguments(EncoderFactory, "encoder_factory")

        parser.add_subclass_arguments(
            RewardFunction, "reward", default=lazy_instance(UnitReward)
        )

        parser.add_argument(
            "--data_layout.root_dir", type=Optional[PathLike], default=None
        )
        parser.add_dataclass_arguments(
            TestSetup,
            "test_setup",
        )
        parser.add_class_arguments(
            InputData, "data_layout.input_data", as_positional=True
        )
        parser.add_dataclass_arguments(OutputData, "data_layout.output_data")

        parser.add_class_arguments(
            ValueEstimatorConfig, as_positional=True, nested_key="estimator_config"
        )
        parser.add_subclass_arguments(
            CriticLoss,
            as_positional=True,
            nested_key="loss",
            skip={"clone_tensordict", "keys"},
        )

        parser.add_class_arguments(
            ActorCritic, "agent", as_positional=True, skip={"keys"}
        )
        parser.add_class_arguments(
            OptimizerSetup, "optimizer_setup", as_positional=True
        )
        parser.add_dataclass_arguments(WandbExtraParameter, "wandb_extra")

        ################################################################################
        #############################                      #############################
        #############################    Link arguments    #############################
        #############################                      #############################
        ################################################################################

        parser.link_arguments(
            "encoder",
            "encoder_factory.encoder_class",
            apply_on="instantiate",
            compute_fn=lambda encoder: encoder.__class__,
        )

        parser.link_arguments(
            "encoder.init_args",
            "encoder_factory.kwargs",
            apply_on="instantiate",
            compute_fn=lambda namespace: vars(namespace),
        )
        parser.link_arguments(
            "encoder_factory",
            "data.encoder_factory",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "encoder.obj_type_id",
            "model.gnn.init_args.obj_type_id",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "encoder.arity_dict",
            "model.gnn.init_args.arity_dict",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "reward",
            "data.reward_function",
            apply_on="instantiate",
            compute_fn=lambda reward: reward,
        )
        parser.link_arguments(
            "estimator_config.gamma",
            "reward.init_args.gamma",
            apply_on="parse",
        )

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
            "data_layout.input_data.domain",
            "encoder.init_args.domain",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data_layout.input_data", "data.input_data", apply_on="instantiate"
        )
        parser.link_arguments(
            "model.gnn.hidden_size", "agent.hidden_size", apply_on="instantiate"
        )

        # Model links
        parser.link_arguments("agent", "model.actor_critic", apply_on="instantiate")

        parser.link_arguments("agent", "optimizer_setup.agent", apply_on="instantiate")
        parser.link_arguments(
            "model.gnn", "optimizer_setup.gnn", apply_on="instantiate"
        )
        parser.link_arguments(
            "optimizer_setup.optimizer", "model.optim", apply_on="instantiate"
        )
        # Loss links
        parser.link_arguments(
            "agent.value_operator",
            "loss.init_args.critic_network",
            apply_on="instantiate",
        )
        parser.link_arguments(
            ("loss", "estimator_config"),
            "model.loss",
            apply_on="instantiate",
            compute_fn=configure_loss,
        )
        parser.link_arguments(
            "estimator_config.estimator_type",
            "model.add_all_rewards_and_done",
            apply_on="instantiate",
            compute_fn=lambda estimator_type: estimator_type
            == "AllActionsValueEstimator",
        )
        parser.link_arguments(
            "estimator_config.estimator_type",
            "model.add_successor_embeddings",
            apply_on="instantiate",
            compute_fn=lambda estimator_type: estimator_type
            == "AllActionsValueEstimator",
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
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.default_root_dir",
            apply_on="instantiate",
        )

        # Validation callback links
        # NOTE it seems like you can't have two callbacks which have different parameter
        # of the same name.
        # Not a problem for dataloader_names as it is the same for all of them.
        parser.link_arguments(
            source="data_layout.input_data",
            target="model.validation_hooks.init_args.dataloader_names",
            compute_fn=validation_dataloader_names,
            apply_on="instantiate",
        )
        # CriticValidation
        parser.link_arguments(
            source="agent.value_operator",
            target="model.validation_hooks.init_args.value_operator",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source=("data_layout.input_data", "reward"),
            target="model.validation_hooks.init_args.discounted_optimal_values",
            compute_fn=discounted_optimal_values_dict,
            apply_on="instantiate",
        )
        # PolicyValidation
        parser.link_arguments(
            source="data_layout.input_data",
            target="model.validation_hooks.init_args.optimal_policy_dict",
            compute_fn=optimal_policy_dict,
            apply_on="instantiate",
        )
        # ProbsStoreCallback
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="model.validation_hooks.init_args.save_dir",
            apply_on="instantiate",
        )

        # PolicyEvaluationValidation
        parser.link_arguments(
            source=("data_layout.input_data", "reward"),
            target="model.validation_hooks.init_args.discounted_optimal_values",
            compute_fn=discounted_optimal_values_dict,
            apply_on="instantiate",
        )
        parser.link_arguments(
            source=("data_layout.input_data.validation_spaces", "reward"),
            target="model.validation_hooks.init_args.envs",
            compute_fn=lambda validation_spaces, reward_func: [
                ExpandedStateSpaceEnv(space, reward_function=reward_func, reset=True)
                for space in validation_spaces
            ],
            apply_on="instantiate",
        )
        parser.link_arguments(
            "estimator_config.gamma",
            "model.validation_hooks.init_args.gamma",
            apply_on="instantiate",
        )

        # ProbsStoreCallback, PolicyEvaluationValidation
        parser.link_arguments(
            source="agent",
            target="model.validation_hooks.init_args.probs_collector",
            compute_fn=lambda agent: ProbsCollector(probs_key=agent.keys.probs),
            apply_on="instantiate",
        )

    def instantiate_trainer(self, **kwargs: Dict) -> Trainer:
        """
        We need to add the validation callbacks of the model to the trainer.
        The problem is that we have a list of callbacks, and we can't extend
        the list of callbacks provided via the config using jsonargparse.
        LightningCLI offers an extra way via "forced callbacks" but that doesn't work with lists.
        Therefore, we manually add our model callbacks to the extra callbacks.
        """
        if (
            isinstance(self.model, PolicyGradientLitModule)
            and self.model.validation_hooks
        ):
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
        if wandb_extra.log_code:  # save everything inside src/rgnet
            self.trainer.logger.experiment.log_code(
                str((Path(__file__) / ".." / ".." / "..").resolve())
            )
