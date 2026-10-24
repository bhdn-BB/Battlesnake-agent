"""Versioned uint8 encoder shared by training, exports, and HTTP inference."""

from collections import Counter

import numpy as np

VERSION = 1
ACTIONS = ("up", "down", "left", "right")
DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))
SHAPE = (31, 11, 11)


def point(p):
    return p["x"], p["y"]


def alive(s):
    return not s.get("eliminated_cause", "") and bool(s["body"])


def slots(board, learner, previous=None):
    ids = [s["id"] for s in board["snakes"] if s["id"] != learner]
    if previous is not None:
        ids = list(previous[1:]) + [i for i in ids if i not in previous]
    return [learner] + ids[:3]


def action_mask(board, learner):
    """Only certain wall/self collisions. Opponent occupancy may disappear on death.

    The pinned engine ignores a snake's body after wall/starvation elimination.
    Thus opponent bodies are deliberately soft features, unless certainty can be
    proved. Old unique tails vacate even on eating; new tails are duplicated.
    """
    you = next(s for s in board["snakes"] if s["id"] == learner)
    body = [point(p) for p in you["body"]]
    occupied = set(body[:-1])
    # These old segments remain after movement for every opponent action. Only
    # hard-mask them if the owner cannot disappear in pre-collision elimination.
    food = {point(p) for p in board["food"]}
    hazards = {point(p) for p in board.get("hazards", [])}
    for s in board["snakes"]:
        if s["id"] == learner or not alive(s):
            continue
        ox, oy = point(s["body"][0])
        certain = True
        for dx, dy in DELTAS:
            target = (ox + dx, oy + dy)
            health = (
                100
                if target in food
                else s["health"] - 1 - (14 if target in hazards else 0)
            )
            if (
                not (
                    0 <= target[0] < board["width"] and 0 <= target[1] < board["height"]
                )
                or health <= 0
            ):
                certain = False
        if certain:
            occupied.update(point(p) for p in s["body"][:-1])
    x, y = body[0]
    mask = np.array(
        [
            0 <= x + dx < board["width"]
            and 0 <= y + dy < board["height"]
            and (x + dx, y + dy) not in occupied
            for dx, dy in DELTAS
        ],
        dtype=bool,
    )
    danger = not mask.any()
    if danger:
        mask[:] = True
    return mask, danger


def encode(board, learner, order=None):
    if (board["width"], board["height"]) != (11, 11):
        raise ValueError("encoder v1 requires an 11x11 Standard board")
    out = np.zeros(SHAPE, np.uint8)
    for channel, key in enumerate(("food", "hazards")):
        for p in board.get(key, []):
            out[channel, p["y"], p["x"]] = 255
    snakes = {s["id"]: s for s in board["snakes"]}
    order = slots(board, learner) if order is None else order
    for k, sid in enumerate(order):
        s = snakes.get(sid)
        if s is None or not alive(s):
            continue
        base = 2 + k * 6
        counts = Counter(point(p) for p in s["body"])
        for i, p in reversed(list(enumerate(s["body"]))):
            x, y = point(p)
            if 0 <= x < 11 and 0 <= y < 11:
                out[base + 1, y, x] = round(255 * (len(s["body"]) - i) / len(s["body"]))
                out[base + 2, y, x] = min(255, counts[(x, y)] * 32)
        x, y = point(s["body"][0])
        out[base, y, x] = 255
        out[base + 3] = round(max(0, s["health"]) * 255 / 100)
        out[base + 4] = min(255, round(len(s["body"]) * 255 / 121))
        out[base + 5] = 255
    out[26] = min(255, board["turn"] * 255 // 1000)
    own = snakes.get(learner)
    if own:
        for k, sid in enumerate(order[1:]):
            s = snakes.get(sid)
            if s and alive(s) and len(s["body"]) >= len(own["body"]):
                x, y = point(s["body"][0])
                for dx, dy in DELTAS:
                    if 0 <= x + dx < 11 and 0 <= y + dy < 11:
                        out[27 + k, y + dy, x + dx] = 255
    out[30] = 255  # explicit valid-board plane
    return out


def webhook_board(payload):
    return {**payload["board"], "turn": payload["turn"]}
