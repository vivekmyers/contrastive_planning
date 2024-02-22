import jax
import jax.numpy as jnp
import haiku as hk
import optax
import collections
from agents.base import Agent
from functools import partial


class LearnedAgent(Agent):
    def __init__(
        self,
        rng,
        repr_dim,
        ds,
        batch_size,
        gamma=0.99,
        use_rotation=False,
        lr=3e-4,
        clip_grad=1.0,
        c=10.0,
        lam_init=1e-3,
    ):
        super().__init__(rng, repr_dim, ds, batch_size)

        self.gamma = gamma
        self.use_rotation = use_rotation
        self.c = c
        self.clip_grad = clip_grad
        self.lr = lr

        x0 = jnp.zeros((1, self.x_dim))
        self.repr_fn = hk.without_apply_rng(hk.transform(self._repr_fn))
        self.params = self.repr_fn.init(next(self.keys), x=x0)
        self.params["log_lambda"] = jnp.log(lam_init)
        self.grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        self.grad_fn = jax.jit(self.grad_fn)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad), optax.adam(learning_rate=lr)
        )
        self.opt_state = self.optimizer.init(self.params)
        self.metrics = collections.defaultdict(list)

    def loss_fn(self, *args, **kwargs):
        raise NotImplementedError

    def _repr_fn(self, x):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def _step_fn(self, opt_state, params, key):
        data = self.get_data(key)
        key, args = data[0], data[1:]
        (loss, metrics), grad = self.grad_fn(params, *args)
        updates, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, metrics, params, opt_state, key

    def step(self):
        key = next(self.keys)
        loss, metrics, self.params, self.opt_state, _ = self._step_fn(
            self.opt_state, self.params, key
        )
        for m in metrics:
            self.metrics[m].append(metrics[m])
        return loss, metrics

    @property
    def _pickle_attrs(self):
        return super()._pickle_attrs + [
            "gamma",
            "use_rotation",
            "c",
            "params",
            "opt_state",
            "clip_grad",
            "lr",
        ]

    def __setstate__(self, state):
        super().__setstate__(state)
        self.repr_fn = hk.without_apply_rng(hk.transform(self._repr_fn))
        self.grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        self.grad_fn = jax.jit(self.grad_fn)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_grad), optax.adam(learning_rate=self.lr)
        )
