"""Frozen scripted pool. Random choices always use the caller's RNG."""

from collections import deque

import numpy as np

from battlesnake.heuristic import choose_move
from env.encoding import ACTIONS, DELTAS, action_mask, alive, point

SCRIPTED = (
    "safe_random",
    "greedy_food",
    "flood_fill",
    "teacher",
    "heuristic",
    "teacher_cautious",
    "flood_hungry",
)
VERSION = "scripted-v1"


def distances(start, blocked, width=11, height=11):
    result = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTAS:
            nxt = x + dx, y + dy
            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and nxt not in blocked
                and nxt not in result
            ):
                result[nxt] = result[(x, y)] + 1
                queue.append(nxt)
    return result


def scripted(name, board, sid, rng):
    variant = name
    if name == "teacher_cautious":
        name = "teacher"
    if name == "flood_hungry":
        name = "flood_fill"
    you = next(s for s in board["snakes"] if s["id"] == sid)
    mask, _ = action_mask(board, sid)
    valid = np.flatnonzero(mask)
    if name == "safe_random":
        return int(rng.choice(valid))
    if name == "heuristic":
        live = [
            {**s, "head": s["body"][0], "length": len(s["body"])}
            for s in board["snakes"]
            if alive(s)
        ]
        return ACTIONS.index(
            choose_move(
                {
                    "board": {**board, "snakes": live},
                    "you": next(s for s in live if s["id"] == sid),
                }
            )
        )
    x, y = point(you["body"][0])
    food = {point(p) for p in board["food"]}
    occupied = {point(p) for s in board["snakes"] if alive(s) for p in s["body"][:-1]}
    threats = {
        (point(s["body"][0])[0] + dx, point(s["body"][0])[1] + dy)
        for s in board["snakes"]
        if alive(s) and s["id"] != sid and len(s["body"]) >= len(you["body"])
        for dx, dy in DELTAS
    }
    scores = []
    for a in valid:
        dx, dy = DELTAS[a]
        target = x + dx, y + dy
        paths = distances(target, occupied, board["width"], board["height"])
        dist = min((paths[p] for p in food if p in paths), default=1000)
        if name == "greedy_food":
            score = -min(
                (abs(target[0] - fx) + abs(target[1] - fy) for fx, fy in food),
                default=0,
            )
        elif name == "flood_fill":
            score = (
                len(paths)
                - 100 * (target in occupied)
                - 15 * (target in threats)
                - (0.5 if variant == "flood_hungry" else 0.01) * dist
            )
        elif name == "teacher":
            space = len(paths)
            length = len(you["body"])
            urgent = max(0, 45 - you["health"]) / 45
            score = (
                min(space, length * 3) * 0.5
                - 200 * (target in occupied)
                - (160 if variant == "teacher_cautious" else 100) * (target in threats)
                - 80 * (space < length)
                - (0.15 + urgent * 3) * dist
                - 150 * (dist >= you["health"])
            )
        else:
            raise ValueError(name)
        scores.append((score, -int(a), int(a)))
    return max(scores)[2]


class EpisodeOpponent:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.hidden = None
        self.order = None
        self.policy = None
        if descriptor not in (*SCRIPTED, "teacher_cautious", "flood_hungry"):
            from training.inference import FrozenPolicy

            self.policy = FrozenPolicy(descriptor)

    def act(self, board, sid, rng):
        if self.policy is None:
            return scripted(self.descriptor, board, sid, rng)
        from env.encoding import slots

        self.order = slots(board, sid, self.order)
        action, self.hidden = self.policy.act(board, sid, self.order, self.hidden)
        return action

    def snapshot(self):
        return dict(descriptor=self.descriptor, hidden=self.hidden, order=self.order)

    @classmethod
    def restore(cls, state):
        instance = cls(state["descriptor"])
        instance.hidden = state["hidden"]
        instance.order = state["order"]
        return instance
