"""Validation selection, paired controls and regression stopping for one model."""

import copy
import json

from env.protocol import EVALUATION, paired_comparison
from training.artifacts import copy_policy, write_json
from training.evaluate import evaluate
from training.inference import export_policy


def score(result):
    return result["win_rate"], -result["timeouts"], -result["latency_ms"]["p95"]


def evaluate_candidate(
    training_session, candidate, folder, split="validation", seed_base=None
):
    return evaluate(
        candidate,
        training_session.config["eval_games"]
        if split == "validation"
        else training_session.config["control_games"],
        split,
        folder,
        max_turns=training_session.config["max_turns"],
        snakes=2,
        opponents=training_session.config["evaluation_opponents"],
        resume=True,
        time_budget_seconds=training_session.remaining_seconds(),
        seed_base=seed_base,
    )


def validate(training_session, candidate, label):
    result = evaluate_candidate(
        training_session, candidate, training_session.output / "validation" / str(label)
    )
    if not result["complete"]:
        return False
    if score(result) > training_session.best:
        training_session.best = score(result)
        copy_policy(candidate, training_session.output / "best.ts")
    previous_peak = training_session.validation_peak
    training_session.regression_count = (
        training_session.regression_count + 1
        if result["win_rate"]
        < previous_peak - training_session.config["regression_threshold"]
        else 0
    )
    training_session.validation_peak = max(previous_peak, result["win_rate"])
    training_session.log(
        event="validation",
        candidate=str(label),
        win_rate=result["win_rate"],
        latency_ms=result["latency_ms"],
        timeouts=result["timeouts"],
        regression_count=training_session.regression_count,
    )
    if not (training_session.output / "champion.ts").exists():
        copy_policy(candidate, training_session.output / "champion.ts")
    return True


def control(training_session, current, regression=False):
    folder = training_session.output / "control" / str(training_session.steps)
    manifest = folder / "candidates.json"
    if manifest.exists():
        candidates = json.loads(manifest.read_text())
    else:
        candidates = {}
        for name, path in (
            ("current", current),
            ("champion", training_session.output / "champion.ts"),
            ("validation_best", training_session.output / "best.ts"),
        ):
            copy_policy(path, folder / f"{name}.ts")
            candidates[name] = f"{name}.ts"
        write_json(manifest, candidates)
    # A separate control set for each event, shared by all three candidates.
    base = EVALUATION["control_seed_base"] + (
        training_session.steps // training_session.config["eval_interval"]
    ) * (training_session.config["control_games"] // 2)
    results = {}
    for name, path in candidates.items():
        result = evaluate_candidate(
            training_session, folder / path, folder / name, "control", base
        )
        if not result["complete"]:
            return False
        results[name] = result
    winner = max(results, key=lambda name: score(results[name]))
    comparison = paired_comparison(results["current"], results["validation_best"])
    confirmed = (
        regression
        and comparison["ci95"][1] < -training_session.config["regression_threshold"]
    )
    write_json(
        folder / "selection.json",
        dict(
            winner=winner,
            regression=regression,
            confirmed_regression=confirmed,
            current_vs_best=comparison,
            comparisons={
                name: paired_comparison(result, results["champion"])
                for name, result in results.items()
            },
            scores={name: score(result) for name, result in results.items()},
        ),
    )
    copy_policy(folder / candidates[winner], training_session.output / "champion.ts")
    training_session.last_control_step = training_session.steps
    training_session.regression_count = 0
    if confirmed:
        training_session.stop_reason = "confirmed_regression"
    training_session.log(
        event="control",
        winner=winner,
        confirmed_regression=confirmed,
        current_vs_best=comparison,
    )
    return True


def publish(training_session, model, vector_env, final=False):
    def export(path):
        if not path.exists():
            export_policy(
                copy.deepcopy(model),
                path,
                "imitation",
                dict(
                    transitions=training_session.steps,
                    seed=training_session.seed,
                    match_snakes=2,
                ),
            )
        return path

    if (
        training_session.steps >= training_session.config["league_start"]
        and training_session.steps - training_session.last_snapshot_step
        >= training_session.config["snapshot_interval"]
    ):
        path = export(
            training_session.output / "league" / f"{training_session.steps}.ts"
        )
        if training_session.latest:
            training_session.pool.append(training_session.latest)
        # At most eight models including the latest snapshot.
        training_session.pool = training_session.pool[-7:]
        training_session.latest = str(path.resolve())
        training_session.last_snapshot_step = training_session.steps
        vector_env.env_method(
            "set_training",
            training_session.steps,
            pool=training_session.pool,
            latest=training_session.latest,
            league=True,
            curriculum=[0, 0],
        )
    current = None
    if (
        training_session.steps - training_session.last_eval_step
        >= training_session.config["eval_interval"]
    ):
        current = export(
            training_session.output / "policies" / f"{training_session.steps}.ts"
        )
        if not validate(training_session, current, training_session.steps):
            return False
        training_session.last_eval_step = training_session.steps
    regression = (
        training_session.regression_count
        >= training_session.config["regression_patience"]
    )
    if training_session.last_control_step != training_session.steps and (
        training_session.steps in training_session.config["control_milestones"]
        or regression
    ):
        current = current or export(
            training_session.output / "policies" / f"{training_session.steps}.ts"
        )
        if not control(training_session, current, regression):
            return False
    if final:
        # Last is mutable across sessions; never overwrite a selected candidate.
        export_policy(
            copy.deepcopy(model),
            training_session.output / "last.ts",
            "imitation",
            dict(
                transitions=training_session.steps,
                seed=training_session.seed,
                match_snakes=2,
            ),
        )
    return True
