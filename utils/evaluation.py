from typing import List
import gym
import jax
import jax.numpy as jnp


import joblib
import numpy as np
import tqdm
from agents import Agent
import d4rl
from utils.training import Dataset, load_data, save_data


def _run_mse_seed(planner: Agent, dataset, support, n_wypt, key):
    traj = dataset.sample_traj(key=key)
    idx = jnp.floor(jnp.arange(1, n_wypt + 1) / (n_wypt + 1) * len(traj)).astype(int)
    obs_w = planner.get_plan(traj[0], traj[-1], n_wypt=n_wypt, support=support)
    err = jnp.sum((obs_w - traj[idx]) ** 2, axis=-1)

    return planner.name, err


def compute_plan_mse(
    loc,
    env,
    planners: List[List[Agent]],
    datasets: List[Dataset],
    support,
    key,
    n_wypt=5,
    trials=100,
    jobs=-1,
):
    mses: dict = {planner_seeds[0].name: [] for planner_seeds in planners}
    keys = jax.random.split(key, trials)

    for name, err in joblib.Parallel(n_jobs=jobs, verbose=1)(
        joblib.delayed(_run_mse_seed)(planner, dataset, support, n_wypt, keys[i])
        for i in tqdm.trange(trials, desc="evaluating plans", disable=True)
        for planner_seeds in planners
        for planner, dataset in zip(planner_seeds, datasets)
    ):
        mses[name].append(err)

    save_data((mses, planners), loc, env, "results")


def compute_rollout_scores(
    loc,
    env,
    planners: List[List[Agent]],
    trials,
    n_wypt,
    datasets: List[Dataset],
    seed=0,
    jobs=-1,
    trunc=5e4,
):

    key = jax.random.key(seed)
    ds_trunc = [ds.truncate(int(trunc)).observations for ds in datasets]
    data = []

    for planner_seeds in planners:
        bins, mu, stderr = eval_planners(
            env, planner_seeds, trials, n_wypt, ds_trunc=ds_trunc, key=key, jobs=jobs
        )
        data.append((planner_seeds, bins, mu, stderr))

    save_data(data, loc, env, "results")


def eval_planners(
    env_name,
    planners: List[Agent],
    trials: int,
    n_wypt: int,
    ds_trunc,
    key,
    max_episode_steps=600,
    jobs=2,
):
    init_l2 = []
    success = []
    keys = jax.random.split(key, len(planners))
    seeds = jax.vmap(jax.random.randint, in_axes=(0, None, None, None))(
        keys, (), 0, jnp.iinfo(jnp.int32).max
    )

    for succ, l2 in joblib.Parallel(n_jobs=jobs, verbose=1)(
        joblib.delayed(_run_rollout)(
            env_name,
            planner,
            seed,
            trials,
            n_wypt,
            max_episode_steps,
            support,
            idx=j,
        )
        for j, (planner, support, seed) in tqdm.tqdm(
            list(enumerate(zip(planners, ds_trunc, seeds))),
            desc="evaluating rollouts",
            disable=True,
        )
    ):
        success.append(succ)
        init_l2.append(l2)

    success = np.array(success)
    init_l2 = np.array(init_l2)
    assert success.shape == init_l2.shape == (len(planners), trials)

    bins = np.linspace(0, 10, 11)
    indices = np.digitize(init_l2, bins)
    avg = np.array(
        [
            [np.mean(success[j][indices[j] == index]) for index in range(len(bins))]
            for j in range(len(planners))
        ]
    )
    assert avg.shape == (len(planners), len(bins))
    mean = np.nanmean(avg, axis=0)
    stderr = np.nanstd(avg, axis=0, ddof=1) / np.sqrt(len(planners))
    valid = ~np.isnan(mean)

    return bins[valid], mean[valid], stderr[valid]


def _run_rollout(
    env_name, planner, seed, trials, n_wypt, max_episode_steps, ds_trunc, idx
):
    env = gym.make(env_name)
    dist = []
    succ = []

    seed = int(seed)
    env.env.reset_target = True
    env.seed(seed)
    np.random.seed(seed)

    for _ in tqdm.trange(
        trials, position=idx % 20, desc=f"seed {idx} ({planner.name})", disable=True
    ):
        s = env.reset()
        goal = env.get_target()
        init_l2 = np.linalg.norm(goal - s[:2])
        done = False
        total_reward = 0
        s_vec = [s]

        if goal.shape != s.shape:
            goal = np.concatenate([goal, 0 * goal], axis=0)
        w_vec = planner.get_plan(s, goal, n_wypt, support=ds_trunc)
        w_vec = np.concatenate([w_vec, goal[None]], axis=0)
        w_index = 0

        for _ in range(max_episode_steps):
            if np.linalg.norm(w_vec[w_index, :2] - s[:2]) < 0.5:
                w_index = min(w_index + 1, len(w_vec) - 1)
            w = w_vec[w_index]
            a = np.sign(w[:2] - (s[:2] + 0.1 * s[2:]))
            s, r, _, done = env.step(a)
            s_vec.append(s)
            total_reward += r

            if total_reward > 0 or done:
                break
        dist.append(init_l2)
        succ.append(total_reward > 0)

    return succ, dist
