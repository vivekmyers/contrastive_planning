import jax
import jax.numpy as jnp
import haiku as hk
import optax
import agents


class NoPlan(agents.Agent):
    name = "none"
    description = "no planning"
    color = "#444444"

    def get_waypoint(self, s, g, n_wypt):
        return jnp.stack([g] * n_wypt, axis=0)

    def __call__(self, x):
        return x
