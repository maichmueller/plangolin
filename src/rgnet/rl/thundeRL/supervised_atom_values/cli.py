import dataclasses
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from lightning.pytorch.cli import OptimizerCallable

# avoids specifying full class_path for model.gnn in cli
from rgnet.models import HeteroGNN, VanillaGNN  # noqa: F401
from rgnet.models.atom_valuator import AtomValuator
from rgnet.rl.thundeRL.cli_config import *
from xmimir import XCategory, XPredicate
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


@dataclasses.dataclass
class CollateKwargs:
    predicates: Optional[Sequence[XPredicate]] = None
    unreachable_atom_value: float = float("inf")


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
    avoid_cycles: bool = False  # whether to avoid cycles during testing


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
        parser.add_class_arguments(CollateKwargs, "collate_kwargs", as_positional=True)

        parser.add_class_arguments(TestSetup, "test_setup", as_positional=True)
        #################################### Validation callback links ##############################

        parser.link_arguments(
            "model.gnn",
            "optimizer_setup.gnn",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "model.atom_valuator", "optimizer_setup.valuator", apply_on="instantiate"
        )
        parser.link_arguments(
            "model.gnn.embedding_size",
            "model.atom_valuator.init_args.feature_size",
            apply_on="parse",
        )
        parser.link_arguments(
            "optimizer_setup.optimizer", "model.optim", apply_on="instantiate"
        )
        parser.link_arguments(
            "data_layout.input_data.domain",
            "model.atom_valuator.init_args.predicates",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "collate_kwargs",
            "data.collate_kwargs",
            apply_on="instantiate",
            compute_fn=lambda collate_kwargs: dataclasses.asdict(collate_kwargs),
        )
        parser.link_arguments(
            "data_layout.input_data.domain",
            "collate_kwargs.predicates",
            apply_on="instantiate",
            compute_fn=lambda domain: [
                XPredicate.make_hollow(
                    category=pred.category, name=pred.name, arity=pred.arity
                )
                for pred in domain.predicates(XCategory.fluent, XCategory.derived)
            ],  # needed to be made hollow so that the collate function can pickle the predicates
        )

        # validation hooks
        parser.link_arguments(
            "data_layout.input_data.domain",
            "model.validation_hooks.init_args.keys",
            apply_on="instantiate",
            compute_fn=lambda domain: [
                pred.name
                for pred in domain.predicates(XCategory.fluent, XCategory.derived)
            ],
        )
