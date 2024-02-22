from typing import List
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from agents import Agent
import sklearn.manifold

from utils.training import Dataset, load_data

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Palatino",
        "figure.titlesize": 20,
    }
)


def plot_random_plan(
    planners: List[List[Agent]],
    datasets: List[Dataset],
    n_wypt: int,
    key,
    trunc,
    fig_path,
    jobs=8,
    suffix="",
):
    key, rng = jax.random.split(key)
    idx = jax.random.randint(rng, (), 0, len(datasets))
    planners_oneseed = [planner_seeds[idx] for planner_seeds in planners]
    return plot_plan(
        planners_oneseed, datasets[idx], n_wypt, key, trunc, fig_path, jobs, suffix
    )


def plot_plan(
    planners: List[Agent],
    ds: Dataset,
    n_wypt: int,
    key,
    trunc,
    fig_path,
    jobs=8,
    suffix="",
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    key, rng1, rng2 = jax.random.split(key, 3)
    start_idx = jax.random.randint(rng1, (), 0, ds.size // 2)
    ds_trunc = ds.drop(start_idx).truncate(trunc)
    seed = jax.random.randint(rng2, (), 0, jnp.iinfo(jnp.int32).max)
    tsne = sklearn.manifold.TSNE(
        n_components=2, random_state=int(seed), verbose=1, n_iter=2000
    )
    try:
        ds_tsne = tsne.fit_transform(ds_trunc.observations)
    except RuntimeError:  # if numerical error occurs, retry
        return plot_plan(planners, ds, n_wypt, key, trunc, fig_path, jobs, suffix)

    ds_obs = np.array(ds_trunc.observations)
    traj = ds_trunc.sample_traj(key)

    for ax, planner in zip(axs.flat, planners):
        idx = np.array(planner.neighbors(traj, ds_obs)).astype(int)
        psi = ds_tsne[idx]

        ax.set_title(planner.description, y=1.05)
        ax.text(psi[0, 0], psi[0, 1], "$x_0$", ha="center", va="bottom", fontsize=16)
        ax.text(psi[-1, 0], psi[-1, 1], "$x_T$", ha="center", va="bottom", fontsize=16)

        ax.plot(psi[:, 0], psi[:, 1], "-", c=planner.color, linewidth=1, alpha=0.1)
        ax.scatter(psi[:, 0], psi[:, 1], c=np.arange(len(psi)), cmap="Blues")
        ax.axis("off")

        ds_plan = planner.get_plan(traj[0], traj[-1], n_wypt=n_wypt, support=ds_obs)
        plan_idx = np.array(planner.neighbors(ds_plan, ds_obs)).astype(int)
        vec = jnp.concatenate([psi[0][None], ds_tsne[plan_idx], psi[-1][None]], axis=0)
        ax.scatter(vec[:, 0], vec[:, 1], c=np.arange(len(vec)), cmap="Reds", zorder=2)
        for x1, x2 in zip(vec, vec[1:]):
            ax.annotate(
                "",
                (x2[0], x2[1]),
                (x1[0], x1[1]),
                arrowprops=dict(arrowstyle="->", color="#888", linewidth=2),
                zorder=1,
                label=planner.description,
            )

    plt.tight_layout()
    save_path = f"{fig_path}/samples/plan_embedding{suffix}.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")


def barplot_mse(loc, env_name, fig_path, title):
    mses, planners = load_data(loc, env_name, "results")
    plt.figure(figsize=(10, 4))

    names = [planner_seeds[0].name for planner_seeds in planners]
    means = [np.mean(mses[name]) for name in names]
    stderrs = [np.std(mses[name], ddof=1) / np.sqrt(len(planners[0])) for name in names]
    colors = [planner_seeds[0].color for planner_seeds in planners]
    with plt.rc_context(
        {
            "figure.titlesize": 28,
            "lines.linewidth": 2,
            "font.size": 18,
        }
    ):
        plt.title(title)
        plt.xlabel("Waypoint MSE")
        plt.barh(
            np.arange(len(planners)),
            means,
            tick_label=[planner_seeds[0].description for planner_seeds in planners],
            color=colors,
            alpha=0.85,
        )
        plt.errorbar(
            means,
            np.arange(len(planners)),
            xerr=stderrs,
            ls="none",
            capsize=8,
            elinewidth=2,
            capthick=2,
            ecolor="k",
        )

        plt.tick_params(
            axis="both",
            which="both",
            left=False,
        )
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_path}/plan_mse.pdf", bbox_inches="tight")
        print(f"Saved to {fig_path}/plan_mse.pdf")


def plot_waypoint_mse(loc, env_name, fig_path, title):
    mses, planners = load_data(loc, env_name, "results")
    plt.figure()
    for planner_seeds in planners:
        planner = planner_seeds[0]
        y_vec = mses[planner.name]
        mu = np.mean(y_vec, axis=0)
        stderr = np.std(y_vec, axis=0, ddof=1) / np.sqrt(len(planner_seeds))
        bins = np.arange(1, len(mu) + 1)
        plt.plot(bins, mu, "-o", c=planner.color, label=planner.description)
        plt.fill_between(bins, mu - stderr, mu + stderr, fc=planner.color, alpha=0.1)

    plt.legend()
    plt.title(title)
    plt.xlabel("Waypoint index")
    plt.ylabel("Waypoint MSE")
    plt.xticks([i for i in range(1, len(mu) + 1)])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{fig_path}/waypoint_mse.pdf", bbox_inches="tight")
    print(f"Saved to {fig_path}/waypoint_mse.pdf")


def plot_rollout_scores(
    loc,
    env,
    fig_path,
    title,
):
    data = load_data(loc, env, "results")

    for planner_seeds, bins, mu, stderr in data:
        planner = planner_seeds[0]
        plt.plot(bins, mu, "-o", c=planner.color, label=planner.description)
        plt.fill_between(bins, mu - stderr, mu + stderr, fc=planner.color, alpha=0.1)

    plt.legend()
    plt.title(title)
    plt.xlabel("Initial L2 Distance to Goal")
    plt.ylabel("Success Rate")
    plt.xlim([1, 10])
    plt.ylim([0, 1])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{fig_path}/rollout_scores.pdf", bbox_inches="tight")
    print(f"Saved to {fig_path}/rollout_scores.pdf")
