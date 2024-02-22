import jax
import jax.numpy as jnp
import haiku as hk
import collections
from utils.training import Dataset


class Agent:
    name = None

    def __init__(self, rng, repr_dim, ds: Dataset, batch_size: int):
        self.keys = hk.PRNGSequence(rng)
        self.repr_dim = repr_dim
        self.ds = ds
        self.x_dim = ds.observations.shape[1]
        self.metrics = collections.defaultdict(list)
        self.batch_size = batch_size

    def __call__(self, x):
        raise NotImplementedError

    def get_data(self, key):
        key, rng1, rng2, rng3 = jax.random.split(key, 4)
        i = jax.random.randint(
            rng1, shape=(self.batch_size,), minval=0, maxval=self.ds.size - 1
        )
        i = i + (i == self.ds.prev_start[i]) - (i == self.ds.next_end[i])
        j = jax.random.randint(
            rng2, shape=(self.batch_size,), minval=self.ds.prev_start[i], maxval=i - 1
        )
        k = jax.random.randint(
            rng3,
            shape=(self.batch_size,),
            minval=i + 1,
            maxval=self.ds.next_end[i],
        )
        s0 = self.ds.observations[j]
        s = self.ds.observations[i]
        ns = self.ds.observations[i + 1]
        g = self.ds.observations[k]
        return key, s0, s, ns, g

    def get_waypoint(self, s, g, n_wypt):
        sg = jnp.array([s, g])
        psi_s, psi_g = self(sg)
        scale = (jnp.arange(1, n_wypt + 1) / (n_wypt + 1))[:, None]
        w = (1 - scale) * psi_s + scale * psi_g
        return w

    def get_plan(self, s, goal, n_wypt, support):
        ds_embed = self(support)
        psi_w = self.get_waypoint(s, goal, n_wypt)
        pdist = jnp.mean((psi_w[:, None] - ds_embed[None]) ** 2, axis=-1)
        indices = jnp.argmin(pdist, axis=1)
        return support[indices]

    def step(self):
        raise NotImplementedError

    def neighbors(self, s, support):
        data = self(s)
        ds_embed = self(support)
        pdist = jnp.mean((data[:, None] - ds_embed[None]) ** 2, axis=-1)
        indices = jnp.argmin(pdist, axis=1)
        return indices

    @property
    def _pickle_attrs(self):
        return ["keys", "repr_dim", "x_dim", "batch_size"]

    def __getstate__(self):
        return {x: getattr(self, x) for x in self._pickle_attrs}

    def __setstate__(self, state):
        for x in self._pickle_attrs:
            setattr(self, x, state[x])
