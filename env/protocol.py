"""Disjoint board seed namespaces and paired evaluation schedules."""

import json
from pathlib import Path

import numpy as np

EVALUATION = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/evaluation.json").read_text(
        encoding="utf-8-sig"
    )
)
OPPONENTS = tuple(EVALUATION["opponents"])
TRAIN_RANGE = (1, 100_000_000)
DEMO_RANGE = (100_000_000, 200_000_000)


def schedule(index, split, opponents=OPPONENTS, seed_base=None):
    pair = index // 2
    base = EVALUATION[f"{split}_seed_base"] if seed_base is None else seed_base
    lo = EVALUATION[f"{split}_seed_base"]
    if not lo <= base + pair < lo + EVALUATION["seed_span"]:
        raise ValueError("evaluation seed outside its split namespace")
    return dict(
        pair=pair,
        slot=index % 2,
        lineup_id=pair % len(opponents),
        lineup=[opponents[pair % len(opponents)]],
        game_seed=base + pair,
    )


def paired_interval(values, seed=2026, samples=5000):
    values = np.asarray(values, dtype=float)
    if len(values) % 2 or not len(values):
        raise ValueError("bootstrap requires complete pairs")
    pairs = values.reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(pairs, len(pairs)).mean() for _ in range(samples)])
    return list(map(float, np.quantile(means, [0.025, 0.975])))


def paired_comparison(left, right):
    a, b = left["rows"], right["rows"]
    keys = ("seed", "slot", "lineup", "pair")
    if len(a) != len(b) or any(any(x[k] != y[k] for k in keys) for x, y in zip(a, b)):
        raise ValueError("paired comparison requires identical games and opponents")
    delta = np.array(
        [int(x["outcome"] == "win") - int(y["outcome"] == "win") for x, y in zip(a, b)]
    )
    return dict(
        win_rate_difference=float(delta.mean()),
        ci95=paired_interval(delta),
        method="bootstrap by board-seed/slot pairs",
    )
