import jax
import jax.numpy as jnp
import sklearn
import agents


class PCAAgent(agents.Agent):
    name = "pca"
    description = "PCA planning"
    color = "#469A66"

    def __init__(self, ds, *args, **kwargs):
        super().__init__(ds=ds, *args, **kwargs)
        self.pca = sklearn.decomposition.PCA(n_components=2)
        self.pca.fit(self.ds.observations)
        self.mean = self.pca.mean_
        self.components = self.pca.components_

    def get_waypoint(self, s, g, n_wypt):
        sg = jnp.array([s, g])
        psi_s, psi_g = (sg - self.pca.mean_) @ self.pca.components_.T

        scale = (jnp.arange(1, n_wypt + 1) / (n_wypt + 1))[:, None]
        w = (1 - scale) * psi_s + scale * psi_g
        return w

    def __call__(self, s):
        return (s - self.pca.mean_) @ self.pca.components_.T

    @property
    def _pickle_attrs(self):
        return super()._pickle_attrs + ["pca", "mean", "components"]
