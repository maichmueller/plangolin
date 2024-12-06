from . import torchrl_patches
from .agents import ActorCritic, EGreedyActorCriticHook, EGreedyModule, EpsilonAnnealing
from .embedding import EmbeddingModule, EmbeddingTransform, NonTensorTransformedEnv
from .losses import ActorCriticLoss
from .optimality_utils import optimal_discounted_values, optimal_policy
from .rollout_collector import RolloutCollector
