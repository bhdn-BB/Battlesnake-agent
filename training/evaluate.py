"""Separate seeded games, slot rotation, game-level bootstrap, and cost accounting."""

import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from env.core import SnakeEnv
from env.opponents import SCRIPTED, scripted
from env.protocol import EVALUATION, OPPONENTS, paired_interval, schedule

from .inference import FrozenPolicy


def win_interval(wins, games):
    """Wilson 95% interval over complete games, including zero/all-win cases."""
    if games < 1:
        raise ValueError("confidence interval needs games")
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    margin = (
        z * math.sqrt((rate * (1 - rate) + z * z / (4 * games)) / games) / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate(
    policy,
    games=200,
    split="validation",
    output=None,
    seed=2026,
    max_turns=1000,
    resume=False,
    time_budget_seconds=None,
    snakes=2,
    seed_base=None,
    opponents=None,
):
    if split not in ("validation", "control", "final"):
        raise ValueError(split)
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    if (
        not opponents
        or len(set(opponents)) != len(opponents)
        or any(p not in SCRIPTED for p in opponents)
    ):
        raise ValueError("opponents must be unique scripted names")
    if games < 2 or games % (2 * len(opponents)):
        raise ValueError("games must balance complete slot pairs across opponents")
    if snakes != 2:
        raise ValueError("evaluation must be strictly 1v1")
    schedule(games - 1, split, opponents, seed_base)
    evaluation_lineups = [[p] for p in opponents]
    from .artifacts import sha256

    started = time.monotonic()
    path = Path(output) if output else None
    if resume and path is None:
        raise ValueError("resumable evaluation needs output")
    spec = dict(
        policy_sha256=sha256(policy) if str(policy) not in SCRIPTED else str(policy),
        games=games,
        split=split,
        seed=seed,
        max_turns=max_turns,
        evaluation=EVALUATION,
    )
    records = []
    if seed_base is not None:
        spec["seed_base"] = seed_base
    game_seed_base = (
        EVALUATION[f"{split}_seed_base"] if seed_base is None else seed_base
    )
    if snakes != 4:
        spec.update(snakes=snakes, lineups=evaluation_lineups)
    if path:
        path.mkdir(parents=True, exist_ok=True)
    if resume:
        spec_path = path / "spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError("evaluation configuration or selected policy changed")
        spec_path.write_text(json.dumps(spec, indent=2))
        journal = path / "games.jsonl"
        if journal.exists():
            # Discard only a torn final line; never accept interior corruption.
            with journal.open("rb+") as stream:
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        stream.truncate(stream.tell() - len(line))
                        break
                    records.append(json.loads(line))
            if [r["row"]["game"] for r in records] != list(range(len(records))):
                raise ValueError("non-contiguous evaluation journal")
        if len(records) > games:
            raise ValueError("evaluation journal exceeds requested games")
        result_path = path / "results.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (
                result.get("complete")
                and result["games"] == games
                and len(records) == games
            ):
                return result
    frozen = FrozenPolicy(policy) if str(policy) not in SCRIPTED else None
    env = SnakeEnv(seed=seed, snakes=snakes, max_turns=max_turns)

    rows = [r["row"] for r in records]
    latencies = [v for r in records for v in r["latencies"]]
    prior_expansions = sum(r["expansions"] for r in records)
    counts = Counter()
    lineups = defaultdict(Counter)
    for row in rows:
        counts[row["outcome"]] += 1
        lineups[str(row["lineup_id"])][row["outcome"]] += 1
    try:
        for i in range(len(rows), games):
            if (
                time_budget_seconds is not None
                and time.monotonic() - started >= time_budget_seconds
            ):
                break
            game = schedule(i, split, opponents, seed_base)
            slot, lineup_id, lineup, game_seed = (
                game[k] for k in ("slot", "lineup_id", "lineup", "game_seed")
            )
            rng = np.random.default_rng([seed, game_seed_base, game["pair"]])
            env.reset(
                seed=game_seed,
                options=dict(
                    snakes=snakes, slot=slot, lineup=lineup, game_seed=game_seed
                ),
            )
            hidden = None
            food = 0
            timeouts = 0
            game_latencies = []
            before_expansions = 0
            trace = [env.board] if i < 3 and output else None
            start = time.monotonic()
            while True:
                t = time.perf_counter()
                if frozen:
                    action, hidden = frozen.act(
                        env.board, env.learner, env.order, hidden
                    )
                else:
                    action = scripted(str(policy), env.board, env.learner, rng)
                latency = (time.perf_counter() - t) * 1000
                latencies.append(latency)
                game_latencies.append(latency)
                timeouts += int(latency > 500)
                _, _, terminated, truncated, info = env.step(action)
                food += info["food"]
                if trace is not None:
                    trace.append(env.board)
                if terminated or truncated:
                    break
            outcome = info["outcome"]
            counts[outcome] += 1
            lineups[str(lineup_id)][outcome] += 1
            row = dict(
                game=i,
                pair=game["pair"],
                seed=game_seed,
                slot=slot,
                lineup=lineup,
                lineup_id=lineup_id,
                outcome=outcome,
                turns=info["turn"],
                cause=info["cause"],
                food=food,
                timeouts=timeouts,
                seconds=time.monotonic() - start,
            )
            rows.append(row)
            if resume:
                record = dict(
                    row=row,
                    latencies=game_latencies,
                    expansions=0 - before_expansions,
                )
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            if trace is not None:
                path = Path(output)
                path.mkdir(parents=True, exist_ok=True)
                (path / f"replay_{i}.json").write_text(json.dumps(trace))
    finally:
        env.close()
    if len(rows) < games:
        result = dict(complete=False, games=len(rows), target_games=games, split=split)
        if path:
            (path / "progress.json").write_text(json.dumps(result, indent=2))
        return result
    wins = np.array([r["outcome"] == "win" for r in rows], np.float32)
    if not len(wins):
        raise ValueError("evaluation needs at least one game")
    result = dict(
        policy=str(policy),
        policy_sha256=spec["policy_sha256"],
        complete=True,
        split=split,
        games=games,
        counts={k: counts[k] for k in ("win", "draw", "loss", "truncation")},
        snakes=snakes,
        win_rate=int(wins.sum()) / games,
        win_rate_ci95=paired_interval(wins),
        ci_method="bootstrap by board-seed/slot pairs, 5000 resamples",
        opponents=list(opponents),
        per_opponent={
            p: summarize([r for r in rows if r["lineup"] == [p]]) for p in opponents
        },
        per_slot={
            str(s): summarize([r for r in rows if r["slot"] == s], paired=False)
            for s in (0, 1)
        },
        per_lineup={k: dict(v) for k, v in lineups.items()},
        survival_mean=float(np.mean([r["turns"] for r in rows])),
        food_mean=float(np.mean([r["food"] for r in rows])),
        elimination_causes=dict(Counter(r["cause"] for r in rows if r["cause"])),
        latency_ms=dict(
            zip(
                ("p50", "p95", "p99"),
                map(float, np.quantile(latencies, [0.5, 0.95, 0.99])),
            )
        ),
        timeouts=sum(r["timeouts"] for r in rows),
        search_expansions=prior_expansions + 0,
        rows=rows,
    )
    if output:
        temporary = Path(output) / "results.json.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        os.replace(temporary, Path(output) / "results.json")
    return result


def summarize(rows, paired=True):
    wins = [int(r["outcome"] == "win") for r in rows]
    return dict(
        games=len(rows),
        win_rate=sum(wins) / len(rows),
        win_rate_ci95=paired_interval(wins)
        if paired
        else win_interval(sum(wins), len(rows)),
        counts=dict(Counter(r["outcome"] for r in rows)),
        elimination_causes=dict(Counter(r["cause"] for r in rows if r["cause"])),
        timeouts=sum(r["timeouts"] for r in rows),
    )
