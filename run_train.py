import argparse
import warnings

import d4rl
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import sklearn.decomposition
import sklearn.manifold
import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
import os

import agents
import utils.training
from utils.training import train, save_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Devices: ", jax.devices())


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="door-human-v0")
parser.add_argument("--num_shuffle", type=int, default=3)
parser.add_argument("--repr_dim", type=int, default=32)
parser.add_argument("--use_rotation", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_steps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()


if __name__ == "__main__":
    key = jax.random.PRNGKey(args.seed)

    key, rng = jax.random.split(key)
    train_dataset, val_dataset = utils.training.get_data(
        args.env, key, shuffle=args.num_shuffle
    )

    contrastive_agent = agents.ContrastiveAgent(
        rng=rng,
        repr_dim=args.repr_dim,
        ds=train_dataset,
        batch_size=args.batch_size,
        gamma=0.9,
        use_rotation=args.use_rotation,
        c=10.0,
    )

    vip_agent = agents.VIPAgent(
        rng=rng,
        repr_dim=args.repr_dim,
        ds=train_dataset,
        batch_size=args.batch_size,
        gamma=0.98,
    )

    pca_agent = agents.PCAAgent(
        rng=rng, repr_dim=args.repr_dim, ds=train_dataset, batch_size=args.batch_size
    )

    no_agent = agents.NoPlan(
        rng=rng, repr_dim=args.repr_dim, ds=train_dataset, batch_size=args.batch_size
    )

    fig_path = f"figures/{args.env}/training/{args.seed}"
    os.makedirs(fig_path, exist_ok=True)
    for agent in [contrastive_agent, vip_agent]:
        utils.training.train(agent, args.train_steps, prefix=f"{fig_path}")

    for agent in [contrastive_agent, vip_agent, pca_agent, no_agent]:
        save_seed(agent, args.env, args.seed, "checkpoints")
    save_seed(train_dataset, args.env, args.seed, "checkpoints")
    save_seed(val_dataset, args.env, args.seed, "checkpoints")
