import dataclasses
from argparse import Namespace
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

# avoids specifying full class_path for encoder in cli
from rgnet.encoding import (  # noqa: F401
    ColorGraphEncoder,
    DirectGraphEncoder,
    GraphEncoderBase,
    HeteroGraphEncoder,
)

# avoids specifying full class_path for model.gnn in cli
from rgnet.models import HeteroGNN, VanillaGNN  # noqa: F401
from rgnet.models.atom_valuator import AtomValuator
from rgnet.rl.data_layout import InputData
from rgnet.rl.losses import (  # noqa: F401
    ActorCriticLoss,
    AllActionsLoss,
    AllActionsValueEstimator,
    CriticLoss,
)
from rgnet.rl.thundeRL.cli_config import ThundeRLCLI
from rgnet.rl.thundeRL.data_module import ThundeRLDataModule

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

from .lit_module import AtomValuesLitModule


class OptimizerSetup:
    def __init__(
        self,
        valuator: AtomValuator,
        gnn: Optional[torch.nn.Module],
        optimizer: OptimizerCallable,  # already partly initialized from cli
        lr_valuator: Optional[float] = None,
        lr_embedding: Optional[float] = None,
    ):
        lr_parameters: List[Dict] = []  # parameter-specific learning rate
        plain_parameter: List = []  # no specific learning rate

        for specific_lr, params in [
            (lr_valuator, valuator.parameters()),
            (lr_embedding, gnn.parameters()),
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
            AtomValuesLitModule,
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
        parser.add_class_arguments(AtomValuator, "atom_valuator", as_positional=True)

        #################################### Validation callback links ##############################

        parser.link_arguments(
            "model.gnn",
            "optimizer_setup.gnn",
        )
        parser.link_arguments(
            "model.atom_valuator",
            "optimizer_setup.valuator",
        )
