"""Teacher data from simulated games; complete game IDs define the split."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from env.core import SnakeEnv
from env.opponents import VERSION, scripted
from env.protocol import DEMO_RANGE
from training.checkpoint import sha256
from training.models import masked


def generate(
    path,
    states=100_000,
    seed=17,
    max_turns=1000,
    progress=None,
    save=None,
    active=None,
    snakes=2,
    randomize_slot=True,
    strong_opponents=True,
):
    if snakes != 2:
        raise ValueError("demonstrations must be 1v1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = SnakeEnv(seed=seed, snakes=snakes, max_turns=max_turns)
    env.configure_training(
        dict(
            randomize_slot=randomize_slot,
            strong_opponents=strong_opponents,
            game_seed_range=DEMO_RANGE,
        )
    )
    rng = np.random.default_rng(seed)
    obs = []
    masks = []
    actions = []
    games = []
    board_seeds = []
    started = time.monotonic()
    game = 0
    transitions = 0
    elapsed = 0.0
    last_save = started
    if progress:
        obs, masks, actions, games = (
            list(progress[k]) for k in ("obs", "masks", "actions", "games")
        )
        board_seeds = list(progress["board_seeds"])
        game, transitions, elapsed = (
            progress[k] for k in ("game", "transitions", "elapsed")
        )
        if progress["env"]:
            env.set_state(progress["env"])
        rng.bit_generator.state = progress["rng"]

    def snapshot():
        return dict(
            obs=obs,
            masks=masks,
            actions=actions,
            games=games,
            board_seeds=board_seeds,
            game=game,
            transitions=transitions,
            elapsed=elapsed + time.monotonic() - started,
            env=env.get_state() if env.snapshot_state else None,
            rng=rng.bit_generator.state,
        )

    try:
        while len(obs) < states:
            if active is not None and not active():
                if save:
                    save(snapshot())
                return None
            state, _ = env.reset()
            board_seeds.append(env.snapshot_state["seed"])
            while True:
                action = scripted("teacher", env.board, env.learner, rng)
                if len(obs) < states:
                    obs.append(state)
                    masks.append(env.action_masks())
                    actions.append(action)
                    games.append(game)
                state, _, terminal, truncated, _ = env.step(action)
                transitions += 1
                if terminal or truncated:
                    break
            game += 1
            if save and time.monotonic() - last_save >= 600:
                save(snapshot())
                last_save = time.monotonic()
    finally:
        env.close()
    ids = np.array(games)
    unique = np.unique(ids)
    rng.shuffle(unique)
    if len(unique) < 2:
        raise ValueError("demonstrations require at least two games")
    validation = np.isin(ids, unique[: max(1, len(unique) // 10)])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            obs=np.stack(obs),
            masks=np.stack(masks),
            actions=np.array(actions),
            games=ids,
            validation=validation,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    manifest = dict(
        board_seeds=board_seeds,
        seed_range=DEMO_RANGE,
        states=len(obs),
        games=game,
        teacher=VERSION,
        snakes=snakes,
        strong_opponents=strong_opponents,
        seed=seed,
        simulated_transitions=transitions,
        seconds=elapsed + time.monotonic() - started,
        sha256=sha256(path),
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, path.with_suffix(".json"))
    return manifest


class BehaviorCloningTrainer:
    """Fit the shared policy to teacher actions with a game-level validation split."""

    def __init__(self, policy, dataset, device, epochs=10, batch_size=256):
        self.policy = policy
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, progress=None, save=None, active=None):
        policy = self.policy
        dataset = self.dataset
        device = self.device
        epochs = self.epochs
        batch_size = self.batch_size
        with np.load(dataset) as archive:
            data = {key: archive[key] for key in archive.files}
        training_indices = np.flatnonzero(~data["validation"])
        validation_indices = np.flatnonzero(data["validation"])
        if not len(training_indices) or not len(validation_indices):
            raise ValueError("empty game-level imitation split")
        if set(data["games"][training_indices]) & set(
            data["games"][validation_indices]
        ):
            raise ValueError("demonstration leakage")
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        best = float("inf")
        weights = None
        history = []
        start_epoch = 0
        if progress:
            optimizer.load_state_dict(progress["optimizer"])
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value):
                        values[key] = value.to(device)
            best, weights, history, training_indices, start_epoch = (
                progress[k] for k in ("best", "weights", "history", "train", "epoch")
            )
        for epoch in range(start_epoch, epochs):
            if active is not None and not active():
                return None
            np.random.shuffle(training_indices)
            policy.train()
            for start in range(0, len(training_indices), batch_size):
                batch_indices = training_indices[start : start + batch_size]
                observations = torch.as_tensor(
                    data["obs"][batch_indices], device=device
                )
                mask = torch.as_tensor(data["masks"][batch_indices], device=device)
                labels = torch.as_tensor(data["actions"][batch_indices], device=device)
                logits = policy.action_net(policy.features_extractor(observations))
                loss = F.cross_entropy(masked(logits, mask), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite cloning loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            policy.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(validation_indices), batch_size):
                    batch_indices = validation_indices[start : start + batch_size]
                    logits = policy.action_net(
                        policy.features_extractor(
                            torch.as_tensor(data["obs"][batch_indices], device=device)
                        )
                    )
                    total += F.cross_entropy(
                        masked(
                            logits,
                            torch.as_tensor(
                                data["masks"][batch_indices], device=device
                            ),
                        ),
                        torch.as_tensor(data["actions"][batch_indices], device=device),
                        reduction="sum",
                    ).item()
            total /= len(validation_indices)
            if not np.isfinite(total):
                raise FloatingPointError("nonfinite cloning validation loss")
            history.append(dict(epoch=epoch + 1, validation_loss=total))
            if total < best:
                best = total
                weights = {
                    k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                }
            if save:
                save(
                    dict(
                        optimizer=optimizer.state_dict(),
                        best=best,
                        weights=weights,
                        history=history,
                        train=training_indices,
                        epoch=epoch + 1,
                    )
                )
        policy.load_state_dict(weights)
        return dict(
            best_validation_loss=best, epochs=history, dataset_sha256=sha256(dataset)
        )
