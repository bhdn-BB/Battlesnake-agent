"""One reproducible teacher imitation -> Masked PPO pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np

from env.bridge import ROOT


def configuration(path=None, overrides=()):
    config = json.loads((ROOT / "configs/duel.json").read_text(encoding="utf-8-sig"))
    if path:
        config.update(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    for item in overrides:
        key, value = item.split("=", 1)
        config[key] = json.loads(value)
    if config["algorithm"] != "imitation" or config["seed"] != 1:
        raise ValueError("the experiment has one imitation → PPO model and seed 1")
    if config["match_snakes"] != 2 or config["evaluation_snakes"] != 2:
        raise ValueError("training and evaluation must be strictly 1v1")
    if not 0 < config["transitions"] <= 10_000_000 or config["hours"] <= 0:
        raise ValueError("invalid budget (maximum 10M transitions)")
    n = config["environments"]
    if n < 1 or config["transitions"] % n:
        raise ValueError("transition budget must be divisible by environments")
    for key in ("checkpoint_interval", "eval_interval", "snapshot_interval"):
        if config[key] <= 0 or config[key] % n:
            raise ValueError(f"{key} must be positive and divisible by environments")
    if n * config["rollout_steps"] < 2 or not 1 <= config["workers"] <= n:
        raise ValueError("invalid vector/rollout size")
    if not 1 <= config["bc_epochs"] <= 15:
        raise ValueError("cloning requires 1–15 epochs")
    if config["league_pool_fraction"] + config["league_latest_fraction"] > 0.3:
        raise ValueError("league cannot exceed 30%")
    return config


def train(config, output, resume=None, stop_after_bc=False):
    from env.vector import BatchedVecEnv
    from training.stage2.trainer import PPOTrainer

    from .checkpoint import TrainingSession

    # Validate even programmatic callers; fresh runs never load legacy weights.
    configuration(overrides=[f"{k}={json.dumps(v)}" for k, v in config.items()])
    if not resume and (Path(output) / "checkpoint.pt").exists():
        raise ValueError("existing run requires --resume")
    if (Path(output) / "final/selection.json").exists():
        raise ValueError("training is frozen after final selection")
    training_session = TrainingSession(config, output, resume)
    vector_env = BatchedVecEnv(
        config["environments"],
        config["workers"],
        config["seed"],
        config["max_turns"],
        config["food_reward"],
    )
    vector_env.env_method("configure_training", config)
    vector_env.env_method("set_training", 0, match_snakes=2, curriculum=[0, 0])
    PPOTrainer(training_session, vector_env).train(stop_after_bc=stop_after_bc)


def export(checkpoint, output):
    import gymnasium as gym
    from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy

    from env.encoding import SHAPE

    from .checkpoint import load_checkpoint
    from .inference import export_policy
    from .models import Extractor, PPOExport

    state = load_checkpoint(checkpoint)
    config = state["config"]
    if config["algorithm"] != "imitation" or config.get("match_snakes") != 2:
        raise ValueError("only new duel imitation checkpoints are supported")
    policy = MaskableActorCriticCnnPolicy(
        gym.spaces.Box(0, 255, SHAPE, np.uint8),
        gym.spaces.Discrete(4),
        lambda _: config["learning_rate"],
        features_extractor_class=Extractor,
        net_arch=[],
        normalize_images=False,
    )
    policy.load_state_dict(state["payload"]["model"])
    export_policy(
        PPOExport(policy),
        output,
        "imitation",
        dict(transitions=state["run"]["steps"], seed=config["seed"], match_snakes=2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", default=str(ROOT / "configs/duel.json"))
    p.add_argument("--output", default="runs/duel_v3/imitation/seed_1")
    p.add_argument("--resume")
    p.add_argument(
        "--stop-after-bc",
        action="store_true",
        help="Save the BC-initialized policy after validation, before PPO collection",
    )
    p.add_argument("--set", action="append", default=[])
    p = commands.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument(
        "--split", choices=["validation", "control", "final"], default="validation"
    )
    p.add_argument("--opponents", nargs="+", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("demos")
    p.add_argument("--output", required=True)
    p.add_argument("--states", type=int, default=150000)
    p = commands.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("report")
    p.add_argument("--runs", default="runs/duel_v3")
    args = parser.parse_args()
    if args.command == "train":
        train(
            configuration(args.config, args.set),
            args.output,
            args.resume,
            stop_after_bc=args.stop_after_bc,
        )
    elif args.command == "export":
        export(args.checkpoint, args.output)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            args.policy,
            args.games,
            args.split,
            args.output,
            resume=args.resume,
            opponents=args.opponents,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    elif args.command == "demos":
        from training.stage1.bc import generate

        print(json.dumps(generate(args.output, args.states, seed=101), indent=2))
    else:
        from training.stage2.experiment import report

        print(json.dumps(report(args.runs), indent=2))


if __name__ == "__main__":
    main()
