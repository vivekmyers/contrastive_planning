import jax
import jax.numpy as jnp
import haiku as hk

import agents


class VIPAgent(agents.LearnedAgent):
    name = "vip"
    description = "VIP planning"
    color = "#009193"

    def loss_fn(self, params, x0, xt, xt1, g):
        psig = self.repr_fn.apply(params, g)
        psi0 = self.repr_fn.apply(params, x0)
        psiT = self.repr_fn.apply(params, xt)
        psiT1 = self.repr_fn.apply(params, xt1)

        eps = 1e-6
        v0 = -jnp.sqrt(jnp.sum(psi0 - psig, axis=-1) ** 2 + eps)
        vt = -jnp.sqrt(jnp.sum(psiT - psig, axis=-1) ** 2 + eps)
        vt1 = -jnp.sqrt(jnp.sum(psiT1 - psig, axis=-1) ** 2 + eps)

        value_loss = (1 - self.gamma) * -v0.mean()
        neg_loss = jnp.logaddexp(
            jax.scipy.special.logsumexp(1 + vt - self.gamma * vt1), -10
        )
        loss = value_loss + neg_loss

        l2 = jax.tree_util.tree_reduce(lambda x, y: x + (y**2).sum(), params, 0.0)
        metrics = dict(loss=loss, value_loss=value_loss, neg_loss=neg_loss, l2=l2)

        return loss, metrics

    def _repr_fn(self, x):
        x = jax.nn.swish(hk.Linear(64)(x))
        for _ in range(5):
            delta = hk.Linear(64)(jax.nn.swish(x))
            x = jax.nn.swish(hk.Linear(64)(delta) + x)
        psi = hk.Linear(self.repr_dim)(x)
        return psi

    def __call__(self, x):
        return self.repr_fn.apply(self.params, x)

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
