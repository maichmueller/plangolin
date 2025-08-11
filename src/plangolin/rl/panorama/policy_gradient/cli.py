import warnings
from itertools import chain
from typing import Callable, Dict, List, Literal, Optional

import torch
from lightning.pytorch.cli import OptimizerCallable
from tensordict.nn import InteractionType

from plangolin.algorithms import bellman_optimal_values, optimal_policy

# avoids specifying full class_path for encoder in cli
from plangolin.encoding import (  # noqa: F401
    ColorGraphEncoder,
    DirectGraphEncoder,
    GraphEncoderBase,
    HeteroGraphEncoder,
)

# avoids specifying full class_path for model.gnn in cli
from plangolin.models import RelationalGNN, VanillaGNN  # noqa: F401
from plangolin.rl.agents import ActorCritic
from plangolin.rl.embedding import EmbeddingModule
from plangolin.rl.losses import (  # noqa: F401
    ActorCriticLoss,
    AllActionsLoss,
    AllActionsValueEstimator,
    CriticLoss,
)
from plangolin.rl.losses.all_actions_estimator import KeyBasedProvider
from plangolin.rl.panorama.cli_config import *
from plangolin.rl.panorama.policy_gradient.lit_module import (
    ActorCriticAgentMaker,
    PolicyGradientLitModule,
)
from plangolin.rl.panorama.utils import wandb_id_resolver

# Import before the cli makes it possible to specify only the class and not the
# full class path for model.validation_hooks in the cli config.
from plangolin.rl.panorama.validation import (  # noqa: F401
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
    hyperparameter["gamma"] = estimator_config.gamma
    hyperparameter["shifted"] = True

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
    exploration_type: InteractionType = InteractionType.MODE
    avoid_cycles: bool = False  # whether to avoid cycles during testing
    iw_search: Optional[IWSearch] = None


@dataclasses.dataclass
class Collater:
    """
    Additional keyword arguments for the collate function.
    Args:
        exclude_keys: Keys to exclude from the collate function.
            (default: None)
        iw_search: The IW search instance to use for the worker.
            (default: None)
    """

    func: Callable
    iw_search: Optional[IWSearch] = None
    encoder_factory: Optional[EncoderFactory] = None
    reward_function: Optional[Callable] = None
    exclude_keys: Optional[List[str]] = None


class CLI(PanoramaCLI):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            PolicyGradientLitModule,
            *args,
            **kwargs,
        )

    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(
            ActorCritic, "agent", as_positional=True, skip={"keys"}
        )
        parser.add_subclass_arguments(
            CriticLoss,
            as_positional=True,
            nested_key="loss",
            skip={"clone_tensordict", "keys"},
        )
        parser.add_class_arguments(
            OptimizerSetup, "optimizer_setup", as_positional=True
        )
        parser.add_class_arguments(
            TestSetup,
            "test_setup",
        )
        parser.add_class_arguments(
            Collater,
            "collate",
        )
        #############################    Link arguments    #############################

        # Model links
        parser.link_arguments("agent", "model.actor_critic", apply_on="instantiate")

        parser.link_arguments("agent", "optimizer_setup.agent", apply_on="instantiate")

        parser.link_arguments(
            "model.gnn.embedding_size", "agent.embedding_size", apply_on="instantiate"
        )
        parser.link_arguments(
            "model.gnn", "optimizer_setup.gnn", apply_on="instantiate"
        )
        parser.link_arguments(
            "optimizer_setup.optimizer", "model.optim", apply_on="instantiate"
        )

        # Loss links
        parser.link_arguments(
            ("loss", "estimator_config"),
            "model.loss",
            apply_on="instantiate",
            compute_fn=configure_loss,
        )
        parser.link_arguments(
            "agent.value_operator",
            "loss.init_args.critic_network",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "reward.deadend_reward",
            "loss.init_args.clamp_min",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "reward.regular_reward",
            "loss.init_args.clamp_max",
            apply_on="instantiate",
            compute_fn=lambda r: 0.0 if r < 0.0 else float("inf"),
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
        parser.link_arguments(
            "encoder_factory",
            "collate.encoder_factory",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "reward",
            "collate.reward_function",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "collate",
            "data.collate_kwargs",
            apply_on="instantiate",
            compute_fn=lambda collate: {
                k: v
                for k, v in dataclasses.asdict(collate).items()
                if v is not None and k != "func"
            },
        )
        parser.link_arguments(
            "collate",
            "data.collate_fn",
            apply_on="instantiate",
            compute_fn=lambda collate: collate.func,
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
                i: bellman_optimal_values(flashdrive.env_aux_data()["pyg_env"])
                for i, flashdrive in enumerate(datasets)
            },
            apply_on="instantiate",
        )
        # PolicyValidation
        parser.link_arguments(
            source="data.validation_datasets",
            target="model.validation_hooks.init_args.optimal_policy_dict",
            compute_fn=lambda datasets: {
                i: optimal_policy(flashdrive.env_aux_data()["pyg_env"])
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
            "reward.gamma",
            "model.validation_hooks.init_args.gamma",
            apply_on="instantiate",
        )

        # ProbsStoreCallback, PolicyEvaluationValidation
        parser.link_arguments(
            source="agent",
            target="model.validation_hooks.init_args.probs_collector",
            compute_fn=lambda agent: ProbsCollector(key=agent.keys.probs),
            apply_on="instantiate",
        )


class EvalCLI(CLI):
    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        # fit subcommand adds this value to the config
        parser.add_class_arguments(
            ActorCriticAgentMaker, "agent_maker", as_positional=True
        )
        parser.add_class_arguments(
            EmbeddingModule,
            "embedding_module",
            as_positional=False,
        )
        parser.link_arguments(
            "data_layout.output_data",
            "trainer.logger.init_args.id",
            compute_fn=wandb_id_resolver,
            apply_on="instantiate",
        )
        parser.link_arguments(
            "model.gnn.init_args.embedding_size",
            "embedding_module.embedding_size",
            apply_on="parse",
        )
        parser.link_arguments(
            "model.gnn",
            "embedding_module.gnn",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "encoder",
            "embedding_module.encoder",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "embedding_module",
            "agent.embedding_module",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "agent",
            "agent_maker.module",
            apply_on="instantiate",
        )
