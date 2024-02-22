import os
import gym
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tqdm
from functools import partial
import pickle


def get_data(env_name, key, split=0.8, shuffle=3):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    ds = Dataset(dataset, "all")
    if shuffle:
        ds = ds.shuffle(key, repeat=shuffle)
    num_train = int(ds.size * split)
    train_ds = ds.truncate(num_train, name="train_dataset")
    val_ds = ds.drop(num_train, name="val_dataset")
    return train_ds, val_ds


class Dataset:
    _fields = [
        "observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
        "prev_start",
        "next_end",
    ]

    def __init__(self, raw_data, name):
        dones = np.logical_or(raw_data["terminals"], raw_data["timeouts"])
        num_tran = raw_data["observations"].shape[0]
        next_end = np.zeros(num_tran)
        prev_start = np.zeros(num_tran)
        future_t = num_tran - 1
        for i in range(num_tran - 1, -1, -1):
            if dones[i]:
                future_t = i
            next_end[i] = future_t
        past_t = 0
        for i in range(num_tran):
            prev_start[i] = past_t
            if dones[i]:
                past_t = i + 1

        aux_data = dict(
            **raw_data,
            next_end=next_end.astype(int),
            prev_start=prev_start.astype(int),
        )
        aux_data = {k: v for k, v in aux_data.items() if k in Dataset._fields}
        aux_data = {
            k: jnp.array(v) if isinstance(v, np.ndarray) else v
            for k, v in aux_data.items()
        }

        self.observations = aux_data["observations"]
        self.actions = aux_data["actions"]
        self.rewards = aux_data["rewards"]
        self.terminals = aux_data["terminals"]
        self.timeouts = aux_data["timeouts"]
        self.prev_start = aux_data["prev_start"]
        self.next_end = aux_data["next_end"]
        self.size = num_tran
        self.name = name

    @classmethod
    def empty(cls):
        new = Dataset.__new__(Dataset)
        for k in Dataset._fields:
            setattr(new, k, jnp.array([]))
        new.size = 0
        new.name = "empty_dataset"
        return new

    def truncate(self, size, name=None):
        new = Dataset.__new__(Dataset)
        for k in Dataset._fields:
            setattr(new, k, getattr(self, k)[:size])
        new.size = size
        new.name = name or f"{self.name}_truncate_{size}"
        return new

    def drop(self, size, name=None):
        new = Dataset.__new__(Dataset)
        for k in Dataset._fields:
            setattr(new, k, getattr(self, k)[size:])
        new.prev_start = jnp.maximum(0, new.prev_start - size)
        new.next_end = jnp.maximum(0, new.next_end - size)
        new.size = self.size - size
        new.name = name or f"{self.name}_drop_{size}"
        return new

    def sample_traj(self, key):
        idx = jax.random.randint(key, (), 0, self.size)
        prev = self.prev_start[idx]
        future = self.next_end[idx]

        return self.observations[prev : future + 1]

    def shuffle(self, key, repeat=3, chunk_size=1000):
        new = self
        for _ in range(repeat):
            key, rng = jax.random.split(key)
            new = self._shuffle(rng, chunk_size)
        return new

    def _shuffle(self, key, chunk_size):
        key, rng = jax.random.split(key)
        start_idx, end_idx = jnp.array(
            [
                [s, e]
                for s, e in tqdm.tqdm(
                    self._chunk(chunk_size, rng),
                    total=int(jnp.sum(self.next_end == jnp.arange(self.size))),
                    desc="chunking data",
                )
                if s != e
            ]
        ).T
        perm = jax.random.permutation(key, len(start_idx))
        raw_data = {
            k: jnp.concatenate(
                [getattr(self, k)[start_idx[i] : end_idx[i]] for i in perm]
            )
            for k in tqdm.tqdm(Dataset._fields, desc="shuffling fields")
            if k != "next_end" and k != "prev_start"
        }
        return Dataset(raw_data, self.name)

    def __len__(self):
        return self.size

    def __add__(self, other):
        new = Dataset.__new__(Dataset)
        for k in Dataset._fields:
            setattr(new, k, jnp.concatenate([getattr(self, k), getattr(other, k)]))
        new.size = self.size + other.size
        new.name = f"{self.name}+{other.name}"
        return new

    def _chunk(self, n, key):
        s = 0
        e = self.next_end[s] + 1
        while e < self.size:
            inter, s, e, key = self._next_chunk(s, e, n, key)
            yield inter
        yield s, e

    @partial(jax.jit, static_argnums=(0,))
    def _next_chunk(self, s, e, n, key):
        key, rng = jax.random.split(key)
        x = jax.random.uniform(rng) < 1 / n
        inter = (s, x * e + (1 - x) * s)
        s = x * e + (1 - x) * s
        e = self.next_end[e] + 1
        return inter, s, e, key


def train(agent, num_steps, prefix="figures", log_interval=20_000):
    os.makedirs(prefix, exist_ok=True)
    for t in tqdm.trange(num_steps + 1):
        agent.step()

        if t % log_interval == 0 and t > 0:
            rows = len(agent.metrics) // 3 + 1
            plt.figure(figsize=(12, rows * 4))
            c_vec = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, (k, v) in enumerate(agent.metrics.items()):
                plt.subplot(3, rows, i + 1)
                v = np.array(v)
                if len(v.shape) == 2:
                    for col in v.T:
                        plt.plot(col)
                    plt.title("%s: %.2f" % (k, v[-1, 0]))
                else:
                    assert len(v.shape) == 1
                    plt.plot(v, alpha=0.1, c=c_vec[0])
                    plt.plot(gaussian_filter1d(v, 50), c=c_vec[0])
                    plt.title("%s: %.2f" % (k, np.mean(v[-500:])))
            plt.tight_layout()
            plt.suptitle(agent.description.capitalize())
            plt.savefig(f"{prefix}/{agent.name}.pdf", bbox_inches="tight")
    return agent


def save_data(data, name, env_name, loc):
    print(f"Saving {name} for {env_name} at {loc}/{env_name}/{name}.pkl")
    os.makedirs(f"{loc}/{env_name}", exist_ok=True)
    pickle.dump(data, open(f"{loc}/{env_name}/{name}.pkl", "wb"))


def load_data(name, env_name, loc):
    print(f"Loading {name} for {env_name}...")
    data = pickle.load(open(f"{loc}/{env_name}/{name}.pkl", "rb"))
    return data


def save_seed(agent, env_name, seed, loc):
    print(
        f"Saving {agent.name} for {env_name} with seed {seed} at {loc}/{env_name}/{seed}/{agent.name}.pkl"
    )
    os.makedirs(f"{loc}/{env_name}/{seed}", exist_ok=True)
    pickle.dump(agent, open(f"{loc}/{env_name}/{seed}/{agent.name}.pkl", "wb"))


def load_seeds(name, env_name, loc):
    print(f"Loading {name} for {env_name}...")
    seeds = os.listdir(f"{loc}/{env_name}")
    agents = []
    for seed in seeds:
        if not os.path.isdir(f"{loc}/{env_name}/{seed}"):
            continue
        agent = pickle.load(open(f"{loc}/{env_name}/{seed}/{name}.pkl", "rb"))
        agents.append(agent)
    print(f"Found {len(agents)} seeds in {loc}/{env_name}")
    return agents
