"""N environments across W CPU processes, one persistent batched engine each."""

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from env.bridge import Engine
from env.core import SnakeEnv


def worker(pipe, indices, kwargs):
    engine = None
    try:
        engine = Engine()
        envs = [
            SnakeEnv(
                seed=kwargs["seed"] + i,
                engine=engine,
                env_id=str(i),
                max_turns=kwargs["max_turns"],
                food_reward=kwargs["food_reward"],
            )
            for i in indices
        ]
        while True:
            cmd, data = pipe.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                result = [e.reset(seed=s)[0] for e, s in zip(envs, data)]
            elif cmd == "step":
                responses = engine.batch([e.request(a) for e, a in zip(envs, data)])
                result = []
                for env, response in zip(envs, responses):
                    obs, reward, terminated, truncated, info = env.finish(response)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated
                    if done:
                        info["terminal_observation"] = obs
                        info["terminal_action_mask"] = env.action_masks()
                        obs = env.reset()[0]
                    result.append((obs, reward, done, info))
            elif cmd == "method":
                name, args, kw, selected = data
                result = [
                    (i, getattr(e, name)(*args, **kw))
                    for i, e in zip(indices, envs)
                    if i in selected
                ]
            elif cmd == "get":
                name, selected = data
                result = [
                    (i, getattr(e, name))
                    for i, e in zip(indices, envs)
                    if i in selected
                ]
            elif cmd == "has":
                name, selected = data
                result = [
                    (i, hasattr(e, name))
                    for i, e in zip(indices, envs)
                    if i in selected
                ]
            elif cmd == "set":
                name, value, selected = data
                for i, e in zip(indices, envs):
                    if i in selected:
                        setattr(e, name, value)
                result = None
            else:
                raise ValueError(cmd)
            pipe.send((True, result))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as exc:
        import traceback

        pipe.send((False, f"{exc}\n{traceback.format_exc()}"))
    finally:
        if engine:
            engine.close()
        pipe.close()


class BatchedVecEnv(VecEnv):
    def __init__(self, count=8, workers=2, seed=1, max_turns=1000, food_reward=0.02):
        if not 1 <= workers <= count:
            raise ValueError("workers must be between 1 and number of environments")
        self.groups = [
            list(map(int, a)) for a in np.array_split(np.arange(count), workers)
        ]
        self.pipes = []
        self.processes = []
        self.closed = False
        self.pending = False
        context = mp.get_context("spawn")
        for group in self.groups:
            parent, child = context.Pipe()
            process = context.Process(
                target=worker,
                args=(
                    child,
                    group,
                    dict(seed=seed, max_turns=max_turns, food_reward=food_reward),
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        import gymnasium as gym

        from env.encoding import SHAPE

        super().__init__(
            count, gym.spaces.Box(0, 255, SHAPE, dtype=np.uint8), gym.spaces.Discrete(4)
        )

    def receive(self, pipe):
        ok, result = pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def reset(self):
        for pipe, group in zip(self.pipes, self.groups):
            pipe.send(("reset", [self._seeds[i] for i in group]))
        obs = [o for pipe in self.pipes for o in self.receive(pipe)]
        self._reset_seeds()
        return np.stack(obs)

    def step_async(self, actions):
        for pipe, group in zip(self.pipes, self.groups):
            pipe.send(("step", [int(actions[i]) for i in group]))
        self.pending = True

    def step_wait(self):
        rows = [r for pipe in self.pipes for r in self.receive(pipe)]
        self.pending = False
        obs, rewards, dones, infos = zip(*rows)
        return (
            np.stack(obs),
            np.array(rewards, np.float32),
            np.array(dones),
            list(infos),
        )

    def _query(self, cmd, data):
        for pipe in self.pipes:
            pipe.send((cmd, data))
        result = [row for pipe in self.pipes for row in (self.receive(pipe) or [])]
        return [value for _, value in sorted(result)]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._query(
            "method",
            (method_name, method_args, method_kwargs, self._get_indices(indices)),
        )

    def get_attr(self, attr_name, indices=None):
        return self._query("get", (attr_name, self._get_indices(indices)))

    def has_attr(self, attr_name):
        return all(self._query("has", (attr_name, self._get_indices(None))))

    def set_attr(self, attr_name, value, indices=None):
        return self._query("set", (attr_name, value, self._get_indices(indices)))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        if self.closed:
            return
        if self.pending:
            self.step_wait()
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe in self.pipes:
            pipe.close()
        self.closed = True
