import dataclasses
from argparse import Namespace
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union, final

import lightning
import torch
from jsonargparse import lazy_instance
from lightning import Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import WandbLogger
from torchrl.objectives import ValueEstimators

# avoids specifying full class_path for encoder in cli
from rgnet.encoding import *  # noqa: F401
from rgnet.encoding.base_encoder import EncoderFactory
from rgnet.logging_setup import get_logger
from rgnet.models import *  # noqa: F401
from rgnet.rl.data import *  # noqa: F401

# avoids specifying full class_path for model.gnn in cli
from rgnet.rl.data_layout import InputData, OutputData
from rgnet.rl.losses import *  # noqa: F401
from rgnet.rl.reward import *  # noqa: F401
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule
from rgnet.rl.thundeRL.validation import *  # noqa: F401

# Import before the cli makes it possible to specify only the class and not the
# full class path for model.validation_hooks in the cli config.
from rgnet.utils.misc import env_aware_cpu_count
from xmimir.iw import IWSearch, IWStateSpace  # noqa: F401,F403


@dataclasses.dataclass
class WandbExtraParameter:
    watch_model: Optional[bool] = True  # whether to watch the model gradients
    log_frequency: int = 100  # the frequency for watch
    log_code: bool = False  # whether to save the code as wandb artifact


def validation_dataloader_names(input_data: InputData) -> Optional[Dict[int, str]]:
    if input_data.validation_problems is None:
        return None

    return {i: p.name for i, p in enumerate(input_data.validation_problems)}


@dataclasses.dataclass
class ValueEstimatorConfig:
    def __init__(
        self,
        gamma: float | None,
        estimator_type: (
            ValueEstimators | Literal["AllActionsValueEstimator"]
        ) | None = ValueEstimators.TD0,
    ):
        self.gamma = gamma
        self.estimator_type = estimator_type


class ThundeRLCLI(LightningCLI):
    def __init__(
        self,
        lit_module_class: Type[lightning.LightningModule],
        lit_data_module_class: Type[lightning.LightningDataModule] = ThundeRLDataModule,
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
            lit_module_class,
            lit_data_module_class,
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

    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        parser.add_subclass_arguments(GraphEncoderBase, "encoder")

        parser.add_class_arguments(EncoderFactory, "encoder_factory")

        parser.add_subclass_arguments(
            RewardFunction, "reward", default=lazy_instance(UnitReward)
        )

        parser.add_argument(
            "--data_layout.root_dir", type=Optional[PathLike], default=None
        )
        parser.add_class_arguments(
            InputData, "data_layout.input_data", as_positional=True
        )
        parser.add_class_arguments(OutputData, "data_layout.output_data")

        parser.add_class_arguments(
            ValueEstimatorConfig, as_positional=True, nested_key="estimator_config"
        )
        parser.add_class_arguments(WandbExtraParameter, "wandb_extra")

        parser.add_argument(
            "--experiment", type=str, required=True, help="Name of the experiment"
        )

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
            apply_on="parse",
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
            "reward.init_args.gamma",
            "estimator_config.gamma",
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

        # Trainer / logger links
        parser.link_arguments(
            source="data_layout.output_data.out_dir",
            target="trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="experiment",
            target="data_layout.output_data.experiment_name",
            apply_on="parse",
        )
        parser.link_arguments(
            source="experiment",
            target="trainer.logger.init_args.name",
            apply_on="parse",
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

    @final
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # while we are not at the root of CLI dependency (ThundeRLCLI), we ask all parents to define their arguments
        # also note: super() is not used here, as we want to call the parent class of `self`, not the
        # parent class of ThundeRLCLI.
        classes = [
            cls
            for cls in type(self).mro()
            if "add_arguments_to_parser_impl" in cls.__dict__
        ]
        # Call implementations from top-most parent down to subclass
        for cls in reversed(classes):
            cls.add_arguments_to_parser_impl(self, parser)

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
        if self.trainer.logger is not None:
            self.trainer.logger.log_hyperparams(
                self.convert_to_nested_dict(self.config["fit"])
            )
            wandb_extra: WandbExtraParameter = self.config_init["fit"]["wandb_extra"]
            if wandb_extra.watch_model and isinstance(self.trainer.logger, WandbLogger):
                self.trainer.logger.watch(
                    self.model, log_freq=wandb_extra.log_frequency
                )
            if wandb_extra.log_code:  # save everything inside src/rgnet
                self.trainer.logger.experiment.log_code(
                    str((Path(__file__) / ".." / ".." / "..").resolve())
                )

    def before_instantiate_classes(self):
        num_threads = self.config.get("data.max_cpu_count")
        limited_num_threads = env_aware_cpu_count()
        num_threads = (
            min(num_threads, limited_num_threads)
            if num_threads
            else limited_num_threads
        )
        torch.set_num_threads(num_threads)
        get_logger(__name__).info(f"[CLI] Set torch cpu threads to {num_threads}.")

    def instantiate_trainer(self, **kwargs: Dict) -> Trainer:
        """
        We need to add the validation callbacks of the model to the trainer.

        The problem is that we have a list of callbacks, and we can't extend
        the list of callbacks provided via the config using jsonargparse.
        LightningCLI offers an extra way via "forced callbacks" but that doesn't work with lists.
        Therefore, we manually add our model callbacks to the extra callbacks.
        """
        if self.model.validation_hooks:
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
