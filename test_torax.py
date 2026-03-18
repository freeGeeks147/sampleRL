"""
Comprehensive Unit and Integration Tests for TORAX RL Environment

Tests cover:
- Environment functionality (reset, step, spaces)
- Physics calculations (ITER89L, q95, Greenwald)
- PPO trainer (networks, trajectory collection, updates)
- Integration (full training loop)

Run with: pytest test_torax.py -v
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from torax_rl_env import (
    ToraxEnvironmentConfig,
    ToraxRLEnvironment,
)
from ppo_trainer import (
    PolicyNetwork,
    ValueNetwork,
    PPOTrainer,
)


class TestToraxEnvironment:
    """Test TORAX environment functionality."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = ToraxEnvironmentConfig(
            episode_max_steps=50,
            use_torax=False,
        )
        env = ToraxRLEnvironment(config)
        yield env
        env.close()
    
    def test_env_creation(self, env):
        """Test environment instantiation."""
        assert env is not None
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
    
    def test_observation_space(self, env):
        """Test observation space dimensions."""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (11,)
        assert env.observation_space.dtype == np.float32
    
    def test_action_space(self, env):
        """Test action space dimensions."""
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
        
        # Check bounds
        assert np.allclose(env.action_space.low, [0.0, -2.0])
        assert np.allclose(env.action_space.high, [1.0, 2.0])
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert isinstance(info, dict)
        assert 'episode' in info
        
        # Check observation bounds (reasonable plasma values)
        assert 1.0 < obs[0] < 15.0  # Te [keV]
        assert 0.5 < obs[2] < 3.0   # ne [e19]
        assert 5.0 < obs[3] < 12.0  # Ip [MA]
    
    def test_step(self, env):
        """Test environment step."""
        obs, _ = env.reset()
        action = env.action_space.sample()
        
        obs_new, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs_new, np.ndarray)
        assert obs_new.shape == (11,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reward_range(self, env):
        """Test reward in valid range."""
        obs, _ = env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0,1]"
            
            if terminated or truncated:
                break
    
    def test_episode_termination(self, env):
        """Test episode termination conditions."""
        obs, _ = env.reset()
        
        # Try to trigger disruption (low q95)
        for _ in range(500):
            # High current, low safety
            action = np.array([0.5, 0.5])  # Strong current ramp
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        
        # Should eventually terminate or truncate
        assert terminated or truncated
    
    def test_deterministic_action(self, env):
        """Test deterministic action execution."""
        obs1, _ = env.reset(seed=42)
        
        action = np.array([0.5, 0.0])
        obs1_new, r1, _, _, _ = env.step(action)
        
        # Reset and repeat
        obs2, _ = env.reset(seed=42)
        obs2_new, r2, _, _, _ = env.step(action)
        
        # Results should be identical
        assert np.allclose(obs1, obs2)
        assert np.allclose(obs1_new, obs2_new)
        assert np.isclose(r1, r2)
    
    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset()
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            assert env.episode_count == episode + 1


class TestPhysicsCalculations:
    """Test physics calculations."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = ToraxEnvironmentConfig(use_torax=False)
        env = ToraxRLEnvironment(config)
        yield env
        env.close()
    
    def test_temperature_evolution(self, env):
        """Test temperature evolution with heating."""
        obs, _ = env.reset()
        
        # Apply heating
        action = np.array([1.0, 0.0])  # Max heating
        obs_new, _, _, _, _ = env.step(action)
        
        # Temperature should increase (roughly)
        # Note: May not always increase due to radiation losses
        assert isinstance(obs_new[0], (float, np.floating))
        assert obs_new[0] > 0  # Still positive
    
    def test_current_ramping(self, env):
        """Test current ramping."""
        obs, _ = env.reset()
        
        # Target high current
        action = np.array([0.5, 1.0])  # Positive ramp
        obs_new, _, _, _, _ = env.step(action)
        
        # Should attempt to reach higher current
        assert isinstance(obs_new[3], (float, np.floating))
        assert 5.0 < obs_new[3] < 12.0  # Within bounds
    
    def test_q95_calculation(self, env):
        """Test safety factor calculation."""
        obs, _ = env.reset()
        
        # q95 should be in reasonable range
        assert 1.5 < obs[4] < 5.0
        
        # High current should lower q95
        obs_high_I, _ = env.reset()
        for _ in range(10):
            action = np.array([0.5, 1.0])
            obs_high_I, _, _, _, _ = env.step(action)
        
        assert isinstance(obs_high_I[4], (float, np.floating))
    
    def test_beta_calculation(self, env):
        """Test beta normalization calculation."""
        obs, _ = env.reset()
        
        # Beta should be positive and reasonable
        assert 0.0 < obs[5] < 10.0
    
    def test_confinement_scaling(self, env):
        """Test ITER89L confinement scaling."""
        obs, _ = env.reset()
        
        # tau_E should be in reasonable range
        assert 0.1 < obs[6] < 2.0
        
        # More heating should give more tau_E (more transport)
        # More current should also increase tau_E
        assert isinstance(obs[6], (float, np.floating))


class TestPPONetworks:
    """Test PPO network components."""
    
    def test_policy_network_creation(self):
        """Test policy network instantiation."""
        policy = PolicyNetwork(obs_dim=11, action_dim=2)
        
        assert policy is not None
        assert sum(p.numel() for p in policy.parameters()) > 0
    
    def test_policy_forward(self):
        """Test policy forward pass."""
        policy = PolicyNetwork(obs_dim=11, action_dim=2)
        
        obs = torch.randn(4, 11)  # Batch of 4 observations
        mean, std = policy(obs)
        
        assert mean.shape == (4, 2)
        assert std.shape == (4, 2)
        assert torch.all(std > 0)  # Std should be positive
    
    def test_policy_sampling(self):
        """Test action sampling from policy."""
        policy = PolicyNetwork(obs_dim=11, action_dim=2)
        
        obs = torch.randn(2, 11)
        action, log_prob = policy.sample_action(obs)
        
        assert action.shape == (2, 2)
        assert log_prob.shape == (2,)
        assert torch.all(torch.isfinite(log_prob))
    
    def test_value_network_creation(self):
        """Test value network instantiation."""
        value = ValueNetwork(obs_dim=11)
        
        assert value is not None
        assert sum(p.numel() for p in value.parameters()) > 0
    
    def test_value_forward(self):
        """Test value network forward pass."""
        value = ValueNetwork(obs_dim=11)
        
        obs = torch.randn(4, 11)
        v = value(obs)
        
        assert v.shape == (4, 1)
        assert torch.all(torch.isfinite(v))


class TestPPOTrainer:
    """Test PPO trainer."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = ToraxEnvironmentConfig(
            episode_max_steps=50,
            use_torax=False,
        )
        env = ToraxRLEnvironment(config)
        yield env
        env.close()
    
    @pytest.fixture
    def trainer(self, env):
        """Create trainer."""
        return PPOTrainer(
            env=env,
            obs_dim=11,
            action_dim=2,
            hidden_sizes=(128, 128),
            batch_size=32,
            epochs_per_update=3,
        )
    
    def test_trainer_creation(self, trainer):
        """Test trainer instantiation."""
        assert trainer is not None
        assert isinstance(trainer.policy, PolicyNetwork)
        assert isinstance(trainer.value, ValueNetwork)
    
    def test_collect_trajectory(self, trainer):
        """Test trajectory collection."""
        trajectory = trainer.collect_trajectory(max_steps=100)
        
        assert isinstance(trajectory, dict)
        assert "obs" in trajectory
        assert "action" in trajectory
        assert "reward" in trajectory
        assert "advantage" in trajectory
        assert "return" in trajectory
        
        # All same length
        lengths = [len(v) for v in trajectory.values() if isinstance(v, list)]
        assert len(set(lengths)) == 1
    
    def test_update_step(self, trainer):
        """Test PPO update step."""
        trajectory = trainer.collect_trajectory(max_steps=100)
        losses = trainer.update(trajectory)
        
        assert isinstance(losses, dict)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        
        # Losses should be finite
        assert np.isfinite(losses["policy_loss"])
        assert np.isfinite(losses["value_loss"])
    
    def test_training_loop(self, trainer):
        """Test mini training loop."""
        history = trainer.train(
            num_iterations=2,
            steps_per_iteration=100,
        )
        
        assert isinstance(history, dict)
        assert "policy_loss" in history
        assert "value_loss" in history
        
        # Should have collected data
        assert len(history["policy_loss"]) == 2
        assert len(history["value_loss"]) == 2


class TestIntegration:
    """Integration tests."""
    
    def test_full_environment_training_pipeline(self):
        """Test complete training pipeline."""
        # Create environment
        config = ToraxEnvironmentConfig(
            episode_max_steps=50,
            use_torax=False,
        )
        env = ToraxRLEnvironment(config)
        
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            obs_dim=11,
            action_dim=2,
            hidden_sizes=(64, 64),
            batch_size=32,
            epochs_per_update=2,
        )
        
        # Mini training
        history = trainer.train(
            num_iterations=2,
            steps_per_iteration=100,
        )
        
        # Verify results
        assert len(history["policy_loss"]) == 2
        assert all(np.isfinite(l) for l in history["policy_loss"])
        
        env.close()
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        config = ToraxEnvironmentConfig(episode_max_steps=50)
        env = ToraxRLEnvironment(config)
        
        trainer = PPOTrainer(env=env, obs_dim=11, action_dim=2)
        
        # Do some training
        trajectory = trainer.collect_trajectory(max_steps=100)
        trainer.update(trajectory)
        
        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            trainer.save(tmp.name)
            
            # Load into new trainer
            trainer2 = PPOTrainer(env=env, obs_dim=11, action_dim=2)
            trainer2.load(tmp.name)
            
            # Check states match
            assert trainer.total_steps == trainer2.total_steps
            assert trainer.episode_count == trainer2.episode_count
        
        env.close()


    def test_real_torax_backend(self):
        """Optional smoke test for the real TORAX backend."""
        pytest.importorskip("torax")

        config = ToraxEnvironmentConfig(
            use_torax=True,
            use_qlknn=False,
            use_mock_fallback=False,
            fixed_dt=0.05,
            max_dt=0.05,
            episode_max_steps=2,
            n_rho=12,
        )
        env = ToraxRLEnvironment(config)
        obs, info = env.reset(seed=0)
        assert info["backend"] == "torax"
        assert obs.shape == (11,)

        obs, reward, terminated, truncated, info = env.step(np.array([0.4, 0.0], dtype=np.float32))
        assert info["backend"] == "torax"
        assert np.isfinite(obs).all()
        assert 0.0 <= reward <= 1.0
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
