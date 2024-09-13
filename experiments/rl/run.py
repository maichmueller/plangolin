import logging
from argparse import ArgumentParser
from datetime import datetime

import torch

from experiments.rl.configs import agent as agent_module
from experiments.rl.configs import data_resolver
from experiments.rl.configs import embedding as embedding_module
from experiments.rl.configs import environment
from experiments.rl.configs import logger as logger_module
from experiments.rl.configs import trainer as trainer_module
from experiments.rl.configs import value_estimator
from rgnet.rl import (
    ActorCritic,
    EmbeddingModule,
    EmbeddingTransform,
    NonTensorTransformedEnv,
)


def create_args():
    parser = ArgumentParser()
    parser = agent_module.add_parser_args(parser)
    parser = environment.add_parser_args(parser)
    parser = embedding_module.add_parser_args(parser)
    parser = logger_module.add_parser_args(parser)
    parser = trainer_module.add_parser_args(parser)
    parser = value_estimator.add_parser_args(parser)
    parser = data_resolver.add_parser_args(parser)
    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=0.9,
        help="Discount factor for the environment (default: 0.9)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        required=False,
        default="auto",
        help="Device to use (default: cuda if available else cpu)",
    )

    return parser


def transformed_environment(
    base_env,
    embedding: EmbeddingModule,
    agent_keys: ActorCritic.AcceptedKeys = ActorCritic.default_keys,
):
    env = NonTensorTransformedEnv(
        env=base_env,
        transform=EmbeddingTransform(
            current_embedding_key=agent_keys.current_embedding,
            env=base_env,
            embedding_module=embedding,
        ),
        cache_specs=True,
        device=embedding.device,
    )
    return env


def run():
    logging.getLogger().setLevel(logging.INFO)
    time_stamp: str = datetime.now().strftime("%d-%m_%H-%M-%S")
    parser = create_args()
    parser_args = parser.parse_args()
    data = data_resolver.from_parser_args(parser_args, exp_id=time_stamp)

    requested_device = parser_args.device
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(requested_device)

    gamma = parser_args.gamma

    base_env = environment.from_parser_args(parser_args, data, device, gamma)

    embedding = embedding_module.from_parser_args(
        parser_args, device=device, data_resolver=data
    )
    env = transformed_environment(
        base_env=base_env,
        embedding=embedding,
        agent_keys=ActorCritic.default_keys,
    )

    agent = agent_module.from_parser_args(
        parser_args, data_resolver=data, embedding=embedding
    )

    value_estimator.from_parser_args(
        parser_args,
        data_resolver=data,
        device=device,
        loss=agent.loss,
        gamma=gamma,
        env=env,
        embedding=embedding,
    )

    logger = logger_module.from_parser_args(
        parser_args, data_resolver=data, agent=agent
    )

    trainer = trainer_module.from_parser_args(
        parser_args, data_resolver=data, logger=logger, agent=agent, env=env
    )
    logging.info(f"Starting training with {len(data.problems)} training problems")
    trainer.train()
    logging.info(f"Finished training. Saved under {data.output_dir}")


if __name__ == "__main__":
    run()
