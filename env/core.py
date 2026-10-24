import copy
import time

import gymnasium as gym
import numpy as np

from env.bridge import Engine
from env.encoding import SHAPE, action_mask, alive, encode, slots
from env.opponents import SCRIPTED, EpisodeOpponent
from env.protocol import TRAIN_RANGE

SETTINGS = dict(minimum_food=1, food_spawn_chance=15, hazard_damage=14)


def transition(before, after, sid, solo=False, food_reward=0.02, max_turns=1000):
    own = next(s for s in after["snakes"] if s["id"] == sid)
    previous = next(s for s in before["snakes"] if s["id"] == sid)
    living = [s for s in after["snakes"] if alive(s)]
    terminated = not alive(own) or (not solo and len(living) <= 1)
    truncated = after["turn"] >= max_turns and not terminated
    outcome = "ongoing"
    reward = food_reward * int(len(own["body"]) > len(previous["body"]))
    if not solo and not living:
        outcome = "draw"
        reward = 0.0
    elif not alive(own):
        outcome = "loss"
        reward = -1.0
    elif terminated:
        outcome = "win"
        reward = 1.0
    elif truncated:
        outcome = "truncation"
    return (
        float(reward),
        terminated,
        truncated,
        dict(
            outcome=outcome, cause=own.get("eliminated_cause", ""), turn=after["turn"]
        ),
    )


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed=1,
        engine=None,
        env_id="0",
        max_turns=1000,
        food_reward=0.02,
        snakes=1,
    ):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, SHAPE, dtype=np.uint8)
        self.engine = engine or Engine()
        self.owns_engine = engine is None
        self.env_id = str(env_id)
        self.max_turns = max_turns
        self.food_reward = food_reward
        self.np_random = np.random.default_rng(seed)
        self.snakes = snakes
        self.match_snakes = None
        self.learner_slot = 0
        self.learner = "s0"
        self.pool = []
        self.latest = None
        self.league = False
        self.danger_count = 0
        self.episode = 0
        self.opponents = {}
        self.snapshot_state = None
        self.training_options = {}
        self.training_transitions = 0

    def configure_training(self, options):
        self.training_options = dict(options)

    @property
    def board(self):
        return self.snapshot_state["board"]

    def set_training(
        self,
        transitions,
        pool=None,
        latest=None,
        league=False,
        curriculum=(100_000, 200_000),
        match_snakes=None,
    ):
        self.training_transitions = transitions
        if match_snakes is not None:
            if match_snakes not in (1, 2, 4):
                raise ValueError("match_snakes must be 1, 2, or 4")
            self.match_snakes = match_snakes
        self.snakes = self.match_snakes or (
            1
            if transitions < curriculum[0]
            else 2
            if transitions < sum(curriculum)
            else 4
        )
        self.pool = pool or []
        self.latest = latest
        self.league = league

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        count = options.get("snakes", self.snakes)
        self.active_snakes = count
        slot = self.learner_slot
        if self.training_options.get("randomize_slot"):
            slot = int(self.np_random.integers(count))
        self.learner = f"s{options.get('slot', slot) % count}"
        game_seed = options.get(
            "game_seed",
            int(
                self.np_random.integers(
                    *self.training_options.get("game_seed_range", TRAIN_RANGE)
                )
            ),
        )
        result = self.engine.call(
            "reset", self.env_id, seed=game_seed, snakes=count, settings=SETTINGS
        )
        self.snapshot_state = result["snapshot"]
        self.order = slots(self.board, self.learner)
        self.opponents = {}
        lineup = options.get("lineup")
        for i, s in enumerate(
            s for s in self.board["snakes"] if s["id"] != self.learner
        ):
            if lineup:
                descriptor = lineup[i]
            else:
                u = self.np_random.random()
                league_ready = (
                    self.league
                    and self.training_transitions
                    >= self.training_options.get("league_start", 0)
                )
                pool_fraction = self.training_options.get("league_pool_fraction", 0.4)
                latest_fraction = self.training_options.get(
                    "league_latest_fraction", 0.2
                )
                if league_ready and u < pool_fraction and self.pool:
                    descriptor = str(self.np_random.choice(self.pool))
                elif (
                    league_ready
                    and pool_fraction <= u < pool_fraction + latest_fraction
                    and self.latest
                ):
                    descriptor = self.latest
                elif self.training_options.get("strong_opponents"):
                    names = [
                        "teacher",
                        "heuristic",
                        "teacher_cautious",
                        "flood_hungry",
                        "greedy_food",
                    ]
                    weights = [0.3, 0.3, 0.2, 0.1, 0.1]
                    descriptor = str(self.np_random.choice(names, p=weights))
                else:
                    descriptor = str(self.np_random.choice(SCRIPTED))
            self.opponents[s["id"]] = EpisodeOpponent(descriptor)
        self.episode += 1
        return self.observation(), {}

    def observation(self):
        return encode(self.board, self.learner, self.order)

    def action_masks(self):
        mask, danger = action_mask(self.board, self.learner)
        self.danger_count += int(danger)
        return mask

    def request(self, action):
        started = time.perf_counter()
        actions = {self.learner: int(action)}
        for sid, opponent in self.opponents.items():
            if alive(next(s for s in self.board["snakes"] if s["id"] == sid)):
                actions[sid] = opponent.act(self.board, sid, self.np_random)
        self.last_opponent_seconds = time.perf_counter() - started
        return dict(op="step", env=self.env_id, actions=actions)

    def finish(self, result):
        before = self.board
        self.snapshot_state = result["snapshot"]
        reward, terminal, truncated, info = transition(
            before,
            self.board,
            self.learner,
            self.active_snakes == 1,
            self.food_reward,
            self.max_turns,
        )
        own = next(s for s in self.board["snakes"] if s["id"] == self.learner)
        previous = next(s for s in before["snakes"] if s["id"] == self.learner)
        info.update(
            unavoidable_danger=self.danger_count,
            food=int(len(own["body"]) > len(previous["body"])),
            snakes=self.active_snakes,
            opponent_seconds=self.last_opponent_seconds,
        )
        return self.observation(), reward, terminal, truncated, info

    def step(self, action):
        return self.finish(self.engine.batch([self.request(action)])[0])

    def get_state(self):
        keys = (
            "snakes",
            "match_snakes",
            "active_snakes",
            "learner",
            "learner_slot",
            "order",
            "pool",
            "latest",
            "league",
            "danger_count",
            "episode",
            "snapshot_state",
            "training_options",
            "training_transitions",
        )
        return dict(
            attributes=copy.deepcopy({k: getattr(self, k) for k in keys}),
            rng=copy.deepcopy(self.np_random.bit_generator.state),
            opponents={k: v.snapshot() for k, v in self.opponents.items()},
        )

    def set_state(self, state):
        self.__dict__.update(copy.deepcopy(state["attributes"]))
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])
        self.opponents = {
            k: EpisodeOpponent.restore(v) for k, v in state["opponents"].items()
        }
        self.engine.call("restore", self.env_id, snapshot=self.snapshot_state)

    def close(self):
        if self.owns_engine:
            self.engine.close()
