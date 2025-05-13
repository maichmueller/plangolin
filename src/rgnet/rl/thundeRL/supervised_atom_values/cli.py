import dataclasses
import warnings
from argparse import Namespace
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import torch
from lightning import Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    OptimizerCallable,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import WandbLogger

from rgnet.algorithms import bellman_optimal_values, optimal_policy

# avoids specifying full class_path for encoder in cli
from rgnet.encoding import (  # noqa: F401
    ColorGraphEncoder,
    DirectGraphEncoder,
    GraphEncoderBase,
    HeteroGraphEncoder,
)

# avoids specifying full class_path for model.gnn in cli
from rgnet.models import HeteroGNN, VanillaGNN  # noqa: F401
from rgnet.rl.agents import ActorCritic
from rgnet.rl.data_layout import InputData
from rgnet.rl.losses import (  # noqa: F401
    ActorCriticLoss,
    AllActionsLoss,
    AllActionsValueEstimator,
    CriticLoss,
)
from rgnet.rl.thundeRL.cli_config import ThundeRLCLI
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule
from rgnet.rl.thundeRL.policy_gradient.lit_module import PolicyGradientLitModule

# Import before the cli makes it possible to specify only the class and not the
# full class path for model.validation_hooks in the cli config.
from rgnet.rl.thundeRL.validation import (  # noqa: F401
    CriticValidation,
    PolicyEntropy,
    PolicyValidation,
    ProbsCollector,
    ProbsStoreCallback,
)
from xmimir.iw import IWSearch, IWStateSpace  # noqa: F401,F403


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


def validation_dataloader_names(input_data: InputData) -> Optional[Dict[int, str]]:
    if input_data.validation_problems is None:
        return None

    return {i: p.name for i, p in enumerate(input_data.validation_problems)}


class AtomValuesCLI(ThundeRLCLI):
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
        )

    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(
            OptimizerSetup, "optimizer_setup", as_positional=True
        )

        #################################### Validation callback links ##############################

        # CriticValidation
        parser.link_arguments(
            source="agent.value_operator",
            target="model.validation_hooks.init_args.value_operator",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data.validation_datasets",
            target="model.validation_hooks.init_args.discounted_optimal_values",
            compute_fn=lambda datasets: {
                i: bellman_optimal_values(flashdrive.env_aux_data.pyg_env)
                for i, flashdrive in enumerate(datasets)
            },
            apply_on="instantiate",
        )
        # PolicyValidation
        parser.link_arguments(
            source="data.validation_datasets",
            target="model.validation_hooks.init_args.optimal_policy_dict",
            compute_fn=lambda datasets: {
                i: optimal_policy(flashdrive.env_aux_data.pyg_env)
                for i, flashdrive in enumerate(datasets)
            },
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
            source="data.validation_datasets",
            target="model.validation_hooks.init_args.envs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "estimator_config.gamma",
            "model.validation_hooks.init_args.gamma",
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
