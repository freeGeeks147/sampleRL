"""Serious-physics PPO baseline for tokamak plasma control.

By default this script uses the real TORAX backend and QLKNN transport model.
A lightweight mock option is still available for rapid debugging, but the
recommended configuration below is the real-physics one.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ppo_trainer import PPOTrainer
from torax_rl_env import ToraxEnvironmentConfig, ToraxRLEnvironment


REAL_PHYSICS_CONFIG = ToraxEnvironmentConfig(
    use_torax=True,
    use_qlknn=True,
    use_mock_fallback=False,
    fixed_dt=0.1,
    max_dt=0.1,
    episode_max_steps=4,
    n_rho=15,
    P_heating_baseline=18.0e6,
    target_T_e=5.5,
    target_n_e=0.85,
    target_q95=5.0,
    target_beta_N=1.2,
    target_tau_E=3.5,
)

FAST_DEV_CONFIG = ToraxEnvironmentConfig(
    use_torax=False,
    use_qlknn=False,
    fixed_dt=0.1,
    episode_max_steps=32,
    target_T_e=8.0,
    target_n_e=1.0,
    target_q95=4.5,
    target_beta_N=1.8,
    target_tau_E=2.5,
)


def build_paper_like_trainer(real_physics: bool = True) -> PPOTrainer:
    """Create the recommended PPO setup.

    Args:
        real_physics: When True, build a TORAX+QLKNN trainer. When False, build
            the fast mock trainer for debugging.
    """
    config = REAL_PHYSICS_CONFIG if real_physics else FAST_DEV_CONFIG
    env = ToraxRLEnvironment(config)
    return PPOTrainer(
        env=env,
        obs_dim=11,
        action_dim=2,
        hidden_sizes=(256, 256),
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.97,
        clip_ratio=0.15,
        entropy_coef=0.01,
        batch_size=16 if real_physics else 128,
        epochs_per_update=4 if real_physics else 10,
    )


def evaluate_policy(trainer: PPOTrainer, episodes: int = 3) -> Dict[str, float]:
    """Run deterministic evaluation episodes and summarize performance."""
    returns = []
    final_q95 = []
    final_beta = []

    for episode in range(episodes):
        obs, _ = trainer.env.reset(seed=episode)
        total_reward = 0.0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = trainer.policy(obs_tensor)
            action = mean.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = trainer.env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)
        final_q95.append(float(obs[4]))
        final_beta.append(float(obs[5]))

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "final_q95_mean": float(np.mean(final_q95)),
        "final_beta_N_mean": float(np.mean(final_beta)),
        "backend": "torax" if trainer.env.config.use_torax else "mock",
    }


def train_and_save(
    output_dir: str = "artifacts/paper_baseline",
    iterations: int = 2,
    steps_per_iteration: int = 16,
    real_physics: bool = True,
) -> Dict[str, float]:
    """Train the recommended PPO baseline and save outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer = build_paper_like_trainer(real_physics=real_physics)
    history = trainer.train(
        num_iterations=iterations,
        steps_per_iteration=steps_per_iteration,
        save_dir=str(output_path),
    )
    metrics = evaluate_policy(trainer)

    trainer.save(str(output_path / "paper_like_policy.pt"))
    config = REAL_PHYSICS_CONFIG if real_physics else FAST_DEV_CONFIG
    (output_path / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_path / "history.json").write_text(
        json.dumps({key: [float(x) for x in values] for key, values in history.items()}, indent=2)
    )
    return metrics


if __name__ == "__main__":
    metrics = train_and_save(real_physics=True)
    print("Paper baseline training complete.")
    print(json.dumps(metrics, indent=2))
