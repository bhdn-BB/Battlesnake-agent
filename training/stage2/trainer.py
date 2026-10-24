"""SB3-Contrib Maskable PPO, optionally initialized by game-split cloning."""

import copy
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecTransposeImage
from torch.nn import functional as F

from training.inference import export_policy
from training.models import Extractor, PPOExport
from training.stage1.bc import BehaviorCloningTrainer, generate
from training.stage2.selection import publish, validate


class StableMaskedPPO(MaskablePPO):
    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        value_clip = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf
            else None
        )
        rows = []
        stopped = False
        batches = 0
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations,
                    data.actions.long().flatten(),
                    action_masks=data.action_masks,
                )
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                log_ratio = log_prob - data.old_log_prob
                ratio = log_ratio.exp()
                policy_loss = -torch.min(
                    advantages * ratio, advantages * ratio.clamp(1 - clip, 1 + clip)
                ).mean()
                prediction = values.flatten()
                if value_clip is not None:
                    prediction = data.old_values + (prediction - data.old_values).clamp(
                        -value_clip, value_clip
                    )
                value_loss = F.mse_loss(data.returns, prediction)
                entropy_loss = (
                    -entropy.mean() if entropy is not None else log_prob.mean()
                )
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    fraction = ((ratio - 1).abs() > clip).float().mean().item()
                if not all(np.isfinite(x) for x in (loss.item(), kl, fraction)):
                    raise FloatingPointError("nonfinite PPO objective/KL")
                row = dict(
                    approx_kl=kl,
                    entropy=-entropy_loss.item(),
                    clip_fraction=fraction,
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                )
                rows.append(row)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    stopped = True
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                row["gradient_norm"] = float(norm)
                row["clipped_gradient_norm"] = min(float(norm), self.max_grad_norm)
                self.policy.optimizer.step()
                batches += 1
                if any(not torch.isfinite(p).all() for p in self.policy.parameters()):
                    raise FloatingPointError("nonfinite PPO weights")
            if stopped:
                break
        self._n_updates += epoch + 1
        ev = float(
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
        )
        # Explained variance is undefined for a constant target; record that explicitly.
        self.diagnostics = {
            k: float(np.mean([r[k] for r in rows if k in r]))
            for k in rows[0]
            if any(k in r for r in rows)
        }
        for key in ("gradient_norm", "clipped_gradient_norm"):
            norms = [r[key] for r in rows if key in r]
            self.diagnostics[key] = float(np.mean(norms)) if norms else 0.0
        self.diagnostics.update(
            explained_variance=ev if np.isfinite(ev) else None,
            explained_variance_defined=np.isfinite(ev).item(),
            kl_stop=stopped,
            optimizer_steps=batches,
            epochs=epoch + 1,
        )


class TrainingCallback(BaseCallback):
    def __init__(self, training_session):
        super().__init__()
        self.training_session = training_session

    def _on_step(self):
        for info in self.locals["infos"]:
            self.training_session.opponent_seconds += info.get("opponent_seconds", 0.0)
            self.training_session.stage_transitions[str(info["snakes"])] += 1
        return True


class PPOTrainer:
    """Execute BC initialization, masked PPO and frozen-opponent self-play."""

    def __init__(self, training_session, vector_env):
        self.training_session = training_session
        self.vector_env = vector_env

    def train(self, stop_after_bc=False):
        training_session = self.training_session
        vector_env = self.vector_env
        config = training_session.config
        # 31 semantic planes exceed the spatial width; SB3's RGB heuristic would
        # incorrectly transpose this already channel-first tensor.
        model = StableMaskedPPO(
            "CnnPolicy",
            VecTransposeImage(vector_env, skip=True),
            learning_rate=config["learning_rate"],
            n_steps=config["rollout_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip"],
            ent_coef=config["entropy"],
            max_grad_norm=config["max_grad_norm"],
            target_kl=config.get("target_kl"),
            clip_range_vf=config.get("value_clip"),
            device=training_session.device,
            seed=training_session.seed,
            policy_kwargs=dict(
                features_extractor_class=Extractor, net_arch=[], normalize_images=False
            ),
            verbose=0,
        )
        _, callback = model._setup_learn(
            config["transitions"],
            callback=TrainingCallback(training_session),
            reset_num_timesteps=True,
        )
        state = training_session.restore(vector_env)
        phase = "demonstrations" if config["algorithm"] == "imitation" else "ppo"
        preparation = None
        if state:
            model.policy.load_state_dict(state["model"])
            model.policy.optimizer.load_state_dict(state["optimizer"])
            model.num_timesteps = training_session.steps
            model._n_updates = training_session.updates
            model._last_obs = state["obs"]
            model._last_episode_starts = state["starts"]
            phase = state.get("phase", "ppo")
            preparation = state.get("preparation")

        def payload():
            return dict(
                model=model.policy.state_dict(),
                optimizer=model.policy.optimizer.state_dict(),
                obs=model._last_obs,
                starts=model._last_episode_starts,
                phase=phase,
                preparation=preparation,
            )

        def save_preparation(progress):
            nonlocal preparation
            preparation = progress
            training_session.save(vector_env, payload())

        try:
            if phase != "ppo":
                dataset = training_session.output / "demonstrations.npz"
                if phase == "demonstrations":
                    if (
                        not dataset.exists()
                        or not dataset.with_suffix(".json").exists()
                    ):
                        generation = generate(
                            dataset,
                            config["demo_states"],
                            training_session.seed + 100,
                            config["max_turns"],
                            preparation,
                            save_preparation,
                            training_session.active,
                            snakes=config.get("match_snakes") or 4,
                            randomize_slot=config.get("randomize_slot", False),
                            strong_opponents=config.get("strong_opponents", False),
                        )
                        if generation is None:
                            return
                        training_session.log(event="demonstrations", **generation)
                    phase, preparation = "cloning", None
                    training_session.save(vector_env, payload())
                if phase == "cloning":
                    result = BehaviorCloningTrainer(
                        model.policy,
                        dataset,
                        training_session.device,
                        config["bc_epochs"],
                        config["batch_size"],
                    ).train(
                        preparation,
                        save_preparation,
                        training_session.active,
                    )
                    if result is None:
                        training_session.save(vector_env, payload())
                        return
                    training_session.log(event="cloning", **result)
                    export_policy(
                        copy.deepcopy(PPOExport(model.policy)),
                        training_session.output / "cloning_only.ts",
                        "imitation",
                        dict(
                            **result,
                            transitions=0,
                            seed=training_session.seed,
                            match_snakes=2,
                        ),
                    )
                    phase, preparation = "cloning_evaluation", None
                    training_session.save(vector_env, payload())
                if not validate(
                    training_session, training_session.output / "cloning_only.ts", 0
                ):
                    training_session.save(vector_env, payload())
                    return
                model.policy.optimizer.state.clear()
                phase, preparation = "ppo", None
                training_session.save(vector_env, payload())
            if stop_after_bc:
                if training_session.steps:
                    raise ValueError(
                        "BC-only mode cannot resume a run that collected PPO steps"
                    )
                return
            callback.on_training_start(locals(), globals())
            # Complete interrupted validation/control before collecting new samples.
            if not publish(training_session, PPOExport(model.policy), vector_env):
                training_session.save(vector_env, payload())
                return
            while training_session.active():
                vector_env.env_method(
                    "set_training",
                    training_session.steps,
                    curriculum=config["curriculum"],
                    pool=training_session.pool,
                    latest=training_session.latest,
                    league=config.get("league", False),
                )
                count = training_session.rollout_count(
                    config["rollout_steps"], vector_env.num_envs
                )
                if count < 1:
                    break
                model.rollout_buffer = MaskableRolloutBuffer(
                    count,
                    vector_env.observation_space,
                    vector_env.action_space,
                    device=training_session.device,
                    gamma=config["gamma"],
                    gae_lambda=config["gae_lambda"],
                    n_envs=vector_env.num_envs,
                )
                model.collect_rollouts(
                    vector_env, callback, model.rollout_buffer, n_rollout_steps=count
                )
                model._update_current_progress_remaining(
                    model.num_timesteps, config["transitions"]
                )
                training_session.steps = model.num_timesteps
                model.train()
                training_session.updates = model._n_updates
                training_session.log(event="ppo_update", **model.diagnostics)
                # Commit learner/optimizer/RNG/env state before potentially long
                # validation or control. A hard stop during evaluation must resume
                # from this policy, without replaying the preceding 50k decisions.
                checkpoint_due = (
                    training_session.steps - training_session.last_save_step
                    >= config["checkpoint_interval"]
                    or time.monotonic() - training_session.last_save_time >= 600
                )
                if checkpoint_due:
                    training_session.save(vector_env, payload())
                if not publish(training_session, PPOExport(model.policy), vector_env):
                    break
                if checkpoint_due:
                    # Also commit selection and league changes made by publish.
                    training_session.save(vector_env, payload())
            # Do not start another evaluation after a session ran out of time.
            if training_session.remaining_seconds() > 0:
                publish(
                    training_session, PPOExport(model.policy), vector_env, final=True
                )
            training_session.save(vector_env, payload())
            callback.on_training_end()
        except (FloatingPointError, RuntimeError) as error:
            if (
                isinstance(error, FloatingPointError)
                or "non-finite" in str(error)
                or "nonfinite" in str(error)
            ):
                training_session.stop_reason = "nonfinite"
                training_session.log(event="numerical_stop", error=str(error))
                training_session.save(vector_env, payload())
            else:
                raise
        finally:
            vector_env.close()
