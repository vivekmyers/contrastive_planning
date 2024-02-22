import argparse
import warnings
import jax
from utils.plotting import (
    plot_waypoint_mse,
    barplot_mse,
    plot_random_plan,
    plot_rollout_scores,
)
from utils.training import load_seeds

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Devices: ", jax.devices())


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="door-human-v0")
parser.add_argument("--n_wypt", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tsne_size", type=int, default=1000)
parser.add_argument("--trials", type=int, default=200)
parser.add_argument("--jobs", type=int, default=10)

parser.add_argument("--barplot", action="store_true")
parser.add_argument("--plot_plan", action="store_true")
parser.add_argument("--waypoint_mse", action="store_true")
parser.add_argument("--rollout", action="store_true")
parser.add_argument(
    "--mse_title", type=str, default="MSE of waypoint prediction for door opening"
)
parser.add_argument("--rollout_title", type=str, default="Maze Planning Success Rates")


args = parser.parse_args()


if __name__ == "__main__":

    fig_path = f"figures/{args.env}"

    contrastive_agents = load_seeds("contrastive", args.env, "checkpoints")
    vip_agents = load_seeds("vip", args.env, "checkpoints")
    pca_agents = load_seeds("pca", args.env, "checkpoints")
    agents = [contrastive_agents, vip_agents, pca_agents]

    train_datasets = load_seeds("train_dataset", args.env, "checkpoints")
    val_datasets = load_seeds("val_dataset", args.env, "checkpoints")

    if args.plot_plan:
        print("Plotting plan TSNE...")
        key = jax.random.key(args.seed)
        plot_random_plan(
            agents,
            val_datasets,
            n_wypt=args.n_wypt,
            trunc=args.tsne_size,
            key=key,
            jobs=args.jobs,
            fig_path=fig_path,
            suffix=args.seed,
        )

    if args.barplot:
        print("Plotting MSE barplot...")
        barplot_mse(
            "plan_mse_1",
            args.env,
            fig_path,
            title=args.mse_title,
        )

    if args.waypoint_mse:
        print("Plotting waypoint MSE...")
        plot_waypoint_mse(
            f"plan_mse_{args.n_wypt}",
            args.env,
            fig_path,
            title=args.mse_title,
        )

    if args.rollout:
        print("Plotting rollouts...")
        plot_rollout_scores(
            "rollout_scores",
            args.env,
            fig_path,
            title=args.rollout_title,
        )
