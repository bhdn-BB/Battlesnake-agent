import copy
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from env.bridge import REVISION, ROOT

from .artifacts import sha256


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    with open(temp, "r+b") as f:
        os.fsync(f.fileno())
    digest = sha256(temp)
    hash_path = path.with_suffix(path.suffix + ".sha256")
    hash_temp = hash_path.with_suffix(hash_path.suffix + ".tmp")
    with hash_temp.open("w", encoding="ascii") as f:
        f.write(digest + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    os.replace(hash_temp, hash_path)
    return digest


def load_checkpoint(path):
    path = Path(path)
    digest = sha256(path)
    hashes = [
        path.with_suffix(path.suffix + ".sha256"),
        path.with_suffix(path.suffix + ".sha256.tmp"),
    ]
    # A process can stop between the two atomic replaces. The already-fsynced
    # pending sidecar verifies that newly committed checkpoint in that window.
    if not any(p.exists() and p.read_text().strip() == digest for p in hashes):
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["revision"] != REVISION:
        raise ValueError("engine revision mismatch")
    return state


def source_hashes():
    paths = []
    for package in ("training", "env"):
        for current, directories, filenames in os.walk(ROOT / package):
            directories[:] = [
                d for d in directories if not d.startswith(".") and d != "__pycache__"
            ]
            paths.extend(
                Path(current) / name for name in filenames if name.endswith(".py")
            )
    paths += list((ROOT / "env/engine").glob("*.go"))
    paths += [ROOT / "training/uv.lock", ROOT / "env/engine/go.sum"]
    paths += list((ROOT / "configs").rglob("*.json"))
    paths += list((ROOT / "battlesnake").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def metadata(config):
    try:
        revision = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    packages = {
        d: importlib.metadata.version(d)
        for d in ("torch", "numpy", "gymnasium", "stable-baselines3", "sb3-contrib")
    }
    return dict(
        config=config,
        engine_revision=REVISION,
        source_revision=revision,
        source_hashes=source_hashes(),
        python=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        cpu=platform.processor(),
        host_memory=psutil.virtual_memory().total,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        started=time.time(),
    )


def memory():
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return dict(
        host_rss_bytes=rss,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
    )


class TrainingSession:
    """Own resumable training state, random generators, budgets and artifacts."""

    def __init__(self, config, output, resume=None):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = config["seed"]
        self.steps = 0
        self.updates = 0
        self.elapsed = 0.0
        self.pool = []
        self.latest = None
        self.best = (-1.0, -float("inf"), -float("inf"))
        self.rng = np.random.default_rng(self.seed)
        self.last_save_step = 0
        self.last_eval_step = 0
        self.last_snapshot_step = 0
        self.stop_reason = None
        self.regression_count = 0
        self.validation_peak = -1.0
        self.last_control_step = -1
        self.expansions = 0
        self.opponent_seconds = 0.0
        self.stage_transitions = {"1": 0, "2": 0, "4": 0}
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(config.get("torch_threads", 1))
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.loaded = load_checkpoint(resume) if resume else None
        if self.loaded:
            previous = self.loaded["config"]
            # Only budgets, output, and CPU worker counts may change on restart.
            allowed = {"transitions", "hours", "workers", "device"}
            if any(
                previous.get(k) != config.get(k)
                for k in set(previous) | set(config)
                if k not in allowed
            ):
                raise ValueError("resume configuration changes the experiment")
            self.__dict__.update(self.loaded["run"])
            self.rng.bit_generator.state = self.loaded["local_rng"]
            self.relocate(self.loaded.get("output"))
        self.started = time.monotonic()
        self.last_save_time = self.started
        self.session_start_steps = self.steps
        metadata_path = self.output / "metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(metadata(config), indent=2), encoding="utf-8"
            )
        (self.output / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def wall_time(self):
        return self.elapsed + time.monotonic() - self.started

    def active(self):
        return (
            self.steps < self.config["transitions"]
            and self.wall_time() < self.config["hours"] * 3600
            and not self.stop_reason
        )

    def remaining_seconds(self):
        return max(0, self.config["hours"] * 3600 - self.wall_time())

    def log(self, **values):
        row = dict(
            transitions=self.steps,
            updates=self.updates,
            wall_seconds=self.wall_time(),
            session_transitions_per_second=(self.steps - self.session_start_steps)
            / max(time.monotonic() - self.started, 1e-9),
            search_expansions=self.expansions,
            opponent_seconds=self.opponent_seconds,
            curriculum_transitions=self.stage_transitions,
            **memory(),
            **values,
        )
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def rollout_count(self, maximum, environments):
        boundaries = [self.config["transitions"]]
        boundaries += [
            s for s in self.config.get("control_milestones", []) if s > self.steps
        ]
        if self.config.get("league_start", 0) > self.steps:
            boundaries.append(self.config["league_start"])
        boundaries += [
            v
            for v in (self.config["curriculum"][0], sum(self.config["curriculum"]))
            if v > self.steps
        ]
        for key in ("checkpoint_interval", "eval_interval", "snapshot_interval"):
            interval = self.config[key]
            boundaries.append((self.steps // interval + 1) * interval)
        return min(maximum, max(1, (min(boundaries) - self.steps) // environments))

    def save(self, vector_env, payload):
        self.last_save_step = self.steps
        self.last_save_time = time.monotonic()
        attrs = (
            "steps",
            "updates",
            "pool",
            "latest",
            "best",
            "last_save_step",
            "last_eval_step",
            "last_snapshot_step",
            "expansions",
            "opponent_seconds",
            "stage_transitions",
            "stop_reason",
            "regression_count",
            "validation_peak",
            "last_control_step",
        )
        # Preserve the serialized "run" key so existing checkpoints stay compatible.
        state = dict(
            version=2,
            output=str(self.output.resolve()),
            revision=REVISION,
            config=self.config,
            run={**{k: getattr(self, k) for k in attrs}, "elapsed": self.wall_time()},
            rng=rng_state(),
            local_rng=self.rng.bit_generator.state,
            envs=vector_env.env_method("get_state"),
            payload=payload,
        )
        atomic_save(state, self.output / "checkpoint.pt")
        self.log(event="checkpoint")
        complete = self.steps == self.config["transitions"]
        status = dict(
            status="stopped"
            if self.stop_reason
            else "trained"
            if complete
            else "incomplete",
            transitions=self.steps,
            target_transitions=self.config["transitions"],
            wall_seconds=self.wall_time(),
            reason=self.stop_reason
            or ("transition_budget" if complete else "session_limit"),
        )
        path = self.output / "status.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(status, indent=2))
        os.replace(temp, path)

    def relocate(self, previous_output):
        """Rebase league references, including opponents in unfinished games."""
        if not previous_output:
            return
        previous = str(previous_output).replace("\\", "/").rstrip("/")

        def path(value):
            if not isinstance(value, str):
                return value
            normalized = value.replace("\\", "/")
            if normalized.startswith(previous + "/"):
                return str(self.output.resolve() / normalized[len(previous) + 1 :])
            return value

        self.pool = [path(p) for p in self.pool]
        self.latest = path(self.latest)
        for name in ("best", "champion", "cloning_only"):
            destination = self.output / f"{name}.ts"
            origin = Path(previous_output) / f"{name}.ts"
            if not destination.exists() and origin.exists():
                from .artifacts import copy_policy

                copy_policy(origin, destination)
        self.loaded["envs"] = copy.deepcopy(self.loaded["envs"])
        references = list(self.pool) + ([self.latest] if self.latest else [])
        for env in self.loaded["envs"]:
            attrs = env["attributes"]
            attrs["pool"] = [path(p) for p in attrs["pool"]]
            attrs["latest"] = path(attrs["latest"])
            references.extend(attrs["pool"])
            if attrs["latest"]:
                references.append(attrs["latest"])
            for opponent in env["opponents"].values():
                opponent["descriptor"] = path(opponent["descriptor"])
                if opponent["descriptor"].endswith(".ts"):
                    references.append(opponent["descriptor"])
        for reference in set(references):
            destination = Path(reference)
            if not destination.exists():
                # Local forked restarts may copy the full bundle automatically.
                relative = destination.relative_to(self.output.resolve())
                origin = Path(previous_output) / relative
                if not origin.exists():
                    raise ValueError(f"missing league artifact: {destination}")
                import shutil

                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin, destination)
                shutil.copy2(
                    origin.with_suffix(".json"), destination.with_suffix(".json")
                )

    def restore(self, vector_env):
        if self.loaded:
            for i, state in enumerate(self.loaded["envs"]):
                vector_env.env_method("set_state", state, indices=i)
            restore_rng(self.loaded["rng"])
            return self.loaded["payload"]
        return None
