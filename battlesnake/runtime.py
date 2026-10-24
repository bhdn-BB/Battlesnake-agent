"""Load the selected duel PPO policy at startup, with heuristic fallback."""

import os
import threading
import time
from pathlib import Path

from .heuristic import choose_move


class PolicyRuntime:
    def __init__(self, path=None, ttl=3600):
        folder = Path(__file__).resolve().parents[1] / "runs/duel_v3/imitation/seed_1"
        default = next(
            (
                folder / name
                for name in ("selected.ts", "champion.ts", "best.ts")
                if (folder / name).exists()
            ),
            None,
        )
        self.path = path or os.getenv("BATTLESNAKE_POLICY") or default
        self.ttl = ttl
        self.games = {}
        self.lock = threading.RLock()
        self.policy = None
        if self.path:
            from training.inference import FrozenPolicy

            self.policy = FrozenPolicy(self.path)

    @staticmethod
    def key(payload):
        return payload["game"]["id"], payload["you"]["id"]

    def start(self, payload):
        with self.lock:
            self.games.pop(self.key(payload), None)

    def end(self, payload):
        with self.lock:
            self.games.pop(self.key(payload), None)

    def move(self, payload):
        if self.policy is None:
            return choose_move(payload)
        if payload.get("game", {}).get("ruleset", {}).get("name", "standard") not in (
            "standard",
            "solo",
        ):
            return choose_move(payload)
        from env.encoding import ACTIONS, slots, webhook_board

        now = time.monotonic()
        key = self.key(payload)
        turn = payload["turn"]
        with self.lock:
            for expired in [
                k for k, v in self.games.items() if now - v["seen"] > self.ttl
            ]:
                del self.games[expired]
            previous = self.games.get(key)
            if previous and turn == previous["turn"]:
                previous["seen"] = now
                return previous["move"]
            # A stale request must not roll a game's recurrent memory backwards.
            if previous and turn < previous["turn"]:
                return previous["move"]
            board = webhook_board(payload)
            sid = payload["you"]["id"]
            order = slots(board, sid, previous["order"] if previous else None)
            hidden = (
                previous["hidden"]
                if previous and turn == previous["turn"] + 1
                else None
            )
            try:
                action, hidden = self.policy.act(board, sid, order, hidden)
                result = ACTIONS[action]
            except (ValueError, RuntimeError, KeyError):
                result = choose_move(payload)
                hidden = None
            self.games[key] = dict(
                turn=turn, move=result, hidden=hidden, order=order, seen=now
            )
            return result

    def close(self):
        with self.lock:
            self.games.clear()
