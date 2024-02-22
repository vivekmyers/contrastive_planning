import argparse
import warnings

import jax
from utils.evaluation import compute_rollout_scores, compute_plan_mse
from utils.training import load_seeds

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Devices: ", jax.devices())


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="door-human-v0")
parser.add_argument("--n_wypt", type=int, default=20)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--jobs", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--mse", action="store_true")
parser.add_argument("--rollout", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    contrastive_agents = load_seeds("contrastive", args.env, "checkpoints")
    vip_agents = load_seeds("vip", args.env, "checkpoints")
    pca_agents = load_seeds("pca", args.env, "checkpoints")
    no_agents = load_seeds("none", args.env, "checkpoints")
    agents = [contrastive_agents, vip_agents, pca_agents, no_agents]

    train_datasets = load_seeds("train_dataset", args.env, "checkpoints")
    val_datasets = load_seeds("val_dataset", args.env, "checkpoints")

    if args.rollout:
        print(f"Generating evaluation rollouts for {args.trials} trials...")
        compute_rollout_scores(
            loc="rollout_scores",
            env=args.env,
            planners=agents,
            n_wypt=args.n_wypt,
            datasets=train_datasets,
            trials=args.trials,
            jobs=args.jobs,
            seed=args.seed,
        )

    if args.mse:
        print("Comparing plan MSEs...")
        compute_plan_mse(
            loc=f"plan_mse_{args.n_wypt}",
            env=args.env,
            planners=agents,
            datasets=val_datasets,
            support=(train_datasets[0] + val_datasets[0]).observations,
            n_wypt=args.n_wypt,
            trials=args.trials,
            key=jax.random.key(args.seed),
            jobs=args.jobs,
        )
