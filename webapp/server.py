"""Run with the training Python: python -m webapp.server."""

import argparse
import hashlib
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from env.bridge import ROOT, Engine
from env.core import SETTINGS, transition
from env.encoding import slots
from env.opponents import scripted
from training.inference import FrozenPolicy

ASSETS = Path(__file__).parent / "static"
DIRECTIONS = ["up", "down", "left", "right"]


class NewGame(BaseModel):
    checkpoint: str
    mode: Literal["play", "watch"] = "play"
    seed: int = Field(
        default_factory=lambda: secrets.randbelow(2**31 - 1) + 1, ge=1, le=2**31 - 1
    )
    slot: int = Field(default=0, ge=0, le=1)


class Turn(BaseModel):
    turn: int = Field(ge=0)


class Move(Turn):
    direction: str


class Arena:
    def __init__(self, models, training_run):
        self.models = Path(models).resolve()
        self.training_run = Path(training_run)
        self.games = {}
        self.lock = threading.RLock()

    def catalog(self):
        entries = []
        for path in self.models.rglob("*"):
            if "arena-cache" in path.relative_to(self.models).parts:
                continue
            if path.suffix == ".ts" and path.with_suffix(".json").is_file():
                try:
                    meta = json.loads(path.with_suffix(".json").read_text())
                except (OSError, ValueError):
                    continue
                if "encoder" not in meta:
                    continue
            elif (
                path.name == "checkpoint.pt"
                and path.with_suffix(".pt.sha256").is_file()
            ):
                try:
                    meta = json.loads((path.parent / "config.json").read_text())
                except (OSError, ValueError):
                    meta = {}
                try:
                    status = json.loads((path.parent / "status.json").read_text())
                    meta["transitions"] = status.get("transitions")
                except (OSError, ValueError):
                    pass
            else:
                continue
            if (
                meta.get("kind", meta.get("algorithm")) != "imitation"
                or meta.get("match_snakes") != 2
            ):
                continue
            # Symlinks cannot turn the model selector into an arbitrary file loader.
            if not path.resolve().is_relative_to(self.models):
                continue
            entries.append(
                dict(
                    id=path.relative_to(self.models).as_posix(),
                    algorithm=meta.get("kind", meta.get("algorithm", "unknown")),
                    transitions=meta.get("transitions"),
                    seed=meta.get("seed"),
                    bytes=path.stat().st_size,
                    modified=path.stat().st_mtime,
                )
            )
        return sorted(entries, key=lambda row: row["modified"], reverse=True)

    def policy_path(self, checkpoint):
        path = (self.models / checkpoint).resolve()
        if not path.is_relative_to(self.models) or checkpoint not in {
            row["id"] for row in self.catalog()
        }:
            raise HTTPException(404, "Checkpoint not found. Refresh the model list.")
        if path.suffix == ".pt":
            from training.cli import export

            stat = path.stat()
            cache = (
                ROOT
                / "runs/arena-cache"
                / str(hashlib.sha256(str(path).encode()).hexdigest()[:16])
                / f"{stat.st_mtime_ns}-{stat.st_size}.ts"
            )
            if not cache.exists():
                export(path, cache)
            return cache
        return path

    def prune(self):
        for key, game in list(self.games.items()):
            if time.monotonic() - game["touched"] > 3600:
                self.games.pop(key)["engine"].close()
        while len(self.games) >= 8:
            key = min(self.games, key=lambda k: self.games[k]["touched"])
            self.games.pop(key)["engine"].close()

    def create(self, request):
        with self.lock:
            if request.mode == "watch" and request.checkpoint == "teacher":
                raise HTTPException(
                    422, "Choose a model checkpoint to watch PPO vs Teacher."
                )
            self.prune()
            frozen = None
            if request.checkpoint != "teacher":
                frozen = FrozenPolicy(self.policy_path(request.checkpoint))
            engine = Engine()
            try:
                state = engine.call(
                    "reset", seed=request.seed, snakes=2, settings=SETTINGS
                )["snapshot"]
                player = f"s{request.slot}"
                opponent = f"s{1 - request.slot}"
                game = dict(
                    engine=engine,
                    snapshot=state,
                    player=player,
                    opponent=opponent,
                    frozen=frozen,
                    hidden=None,
                    order=slots(state["board"], opponent),
                    rng=np.random.default_rng(request.seed),
                    done=False,
                    outcome="ongoing",
                    touched=time.monotonic(),
                    checkpoint=request.checkpoint,
                    mode=request.mode,
                    teacher_move=None,
                    model_move=None,
                    model_latency_ms=None,
                )
                key = secrets.token_urlsafe(20)
                self.games[key] = game
                return self.view(key, game)
            except Exception:
                engine.close()
                raise

    def view(self, key, game):
        return dict(
            id=key,
            board=game["snapshot"]["board"],
            seed=game["snapshot"]["seed"],
            player=game["player"],
            opponent=game["opponent"],
            done=game["done"],
            outcome=game["outcome"],
            checkpoint=game["checkpoint"],
            mode=game["mode"],
            teacher_move=game["teacher_move"],
            model_move=game["model_move"],
            model_latency_ms=game["model_latency_ms"],
        )

    def step(self, key, request, watch=False):
        if not watch and request.direction not in DIRECTIONS:
            raise HTTPException(422, "Unknown move direction.")
        with self.lock:
            game = self.games.get(key)
            if game is None:
                raise HTTPException(
                    404, "Game not found or session closed. Start a new duel."
                )
            if (game["mode"] == "watch") != watch:
                raise HTTPException(409, "This action does not match the game mode.")
            if game["done"] or request.turn != game["snapshot"]["board"]["turn"]:
                raise HTTPException(
                    409, "This turn has already been played or the game is over."
                )
            before = game["snapshot"]["board"]
            model_latency_ms = None
            # Both decisions use the same board, before either move is resolved.
            if game["frozen"]:
                started = time.perf_counter_ns()
                action, hidden = game["frozen"].act(
                    before, game["opponent"], game["order"], game["hidden"]
                )
                model_latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                game["hidden"] = hidden
            else:
                action = scripted("teacher", before, game["opponent"], game["rng"])
            player_action = (
                scripted("teacher", before, game["player"], game["rng"])
                if watch
                else DIRECTIONS.index(request.direction)
            )
            result = game["engine"].call(
                "step",
                actions={
                    game["player"]: player_action,
                    game["opponent"]: action,
                },
            )
            game["snapshot"] = result["snapshot"]
            _, terminal, truncated, info = transition(
                before,
                game["snapshot"]["board"],
                game["opponent"] if watch else game["player"],
                max_turns=1000,
            )
            game.update(
                done=terminal or truncated,
                outcome=info["outcome"],
                model_move=DIRECTIONS[action],
                model_latency_ms=model_latency_ms,
                teacher_move=DIRECTIONS[player_action] if watch else None,
                touched=time.monotonic(),
            )
            return self.view(key, game)

    def close(self):
        for game in self.games.values():
            game["engine"].close()
        self.games.clear()


def create_app(
    models=ROOT / "runs/duel_v3", training_run=ROOT / "runs/duel_v3/imitation/seed_1"
):
    arena = Arena(models, training_run)

    @asynccontextmanager
    async def lifespan(app):
        yield
        arena.close()

    app = FastAPI(title="Battlesnake Duel", lifespan=lifespan)
    app.state.arena = arena

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/checkpoints")
    def checkpoints():
        return dict(checkpoints=arena.catalog())

    @app.get("/api/training")
    def training():
        result = {}
        # A fresh private session can be running before its first downloadable
        # checkpoint. Report that state instead of saying training never began.
        if (
            arena.training_run.resolve()
            == (ROOT / "runs/duel_v3/imitation/seed_1").resolve()
        ):
            try:
                state = json.loads(
                    (ROOT / "runs/kaggle-duel-v3/controller.json").read_text()
                )
                result["remote"] = {
                    key: state.get(key)
                    for key in ("status", "worker_status", "checked_unix", "active")
                }
            except (OSError, ValueError):
                result["remote"] = None
        for name in ("status", "config"):
            try:
                result[name] = json.loads(
                    (arena.training_run / f"{name}.json").read_text()
                )
            except (OSError, ValueError):
                result[name] = None
        try:
            # Only read the tail; long training logs can grow large.
            with (arena.training_run / "metrics.jsonl").open("rb") as stream:
                stream.seek(0, 2)
                offset = max(0, stream.tell() - 32768)
                stream.seek(offset)
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            if offset:
                lines = lines[1:]
            result["metrics"] = [json.loads(line) for line in lines if line.strip()][
                -10:
            ]
        except (OSError, ValueError):
            result["metrics"] = []
        return result

    @app.post("/api/games")
    def new_game(request: NewGame):
        try:
            return arena.create(request)
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(400, f"Could not load the model: {error}") from error

    @app.post("/api/games/{key}/moves")
    def move(key: str, request: Move):
        return arena.step(key, request)

    @app.post("/api/games/{key}/watch")
    def watch(key: str, request: Turn):
        return arena.step(key, request, watch=True)

    @app.delete("/api/games/{key}")
    def delete_game(key: str):
        with arena.lock:
            game = arena.games.pop(key, None)
            if game:
                game["engine"].close()
        return dict(closed=True)

    app.mount("/static", StaticFiles(directory=ASSETS), name="static")
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Play a local 1v1 against a checkpoint"
    )
    parser.add_argument("--models", type=Path, default=ROOT / "runs/duel_v3")
    parser.add_argument(
        "--training-run", type=Path, default=ROOT / "runs/duel_v3/imitation/seed_1"
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.models, args.training_run), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
