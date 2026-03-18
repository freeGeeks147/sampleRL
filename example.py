"""
Complete Working Example - TORAX RL Environment with PPO Training

This script demonstrates:
1. Creating and interacting with the TORAX RL environment
2. Training a PPO agent to control plasma
3. Evaluating the trained policy

Run directly: python example.py
"""

import numpy as np
import torch
import logging
from torax_rl_env import ToraxEnvironmentConfig, ToraxRLEnvironment
from ppo_trainer import PPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_environment():
    """Example 1: Basic environment interaction."""
    print("\n" + "="*80)
    print("Example 1: Basic Environment Interaction")
    print("="*80)
    
    # Create environment
    config = ToraxEnvironmentConfig(
        t_final=5.0,
        episode_max_steps=50,
        use_torax=False,  # Use mock physics for speed
    )
    
    env = ToraxRLEnvironment(config)
    
    # Reset and interact
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Take 5 random steps
    print("\nTaking 5 random actions:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: Action={action}, Reward={reward:.3f}")
        print(f"           Te={obs[0]:.1f}keV, ne={obs[2]:.1f}e19, q95={obs[4]:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✓ Example 1 Complete\n")


def example_2_qlknn_transport():
    """Example 2: Transport model verification."""
    print("="*80)
    print("Example 2: Transport Model")
    print("="*80)
    
    config = ToraxEnvironmentConfig()
    env = ToraxRLEnvironment(config)
    
    # Run a full episode and track transport
    obs, _ = env.reset()
    print(f"\nRunning episode with fixed heating (80% ECRH)...")
    
    rewards = []
    obs_history = []
    
    for step in range(30):
        action = np.array([0.8, 0.0])  # 80% heating, no current ramp
        obs, reward, terminated, truncated, _ = env.step(action)
        
        rewards.append(reward)
        obs_history.append(obs.copy())
        
        if terminated or truncated:
            break
    
    # Analyze trajectory
    obs_array = np.array(obs_history)
    print(f"\nTrajectory Analysis ({len(obs_history)} steps):")
    print(f"  Te range: {obs_array[:, 0].min():.1f} → {obs_array[:, 0].max():.1f} keV")
    print(f"  ne range: {obs_array[:, 2].min():.1f} → {obs_array[:, 2].max():.1f} e19")
    print(f"  q95 range: {obs_array[:, 4].min():.2f} → {obs_array[:, 4].max():.2f}")
    print(f"  Reward range: {np.min(rewards):.3f} → {np.max(rewards):.3f}")
    print(f"  Mean reward: {np.mean(rewards):.3f}")
    
    env.close()
    print("\n✓ Example 2 Complete\n")


def example_3_mini_training():
    """Example 3: Mini training loop (3 episodes for quick test)."""
    print("="*80)
    print("Example 3: Mini PPO Training (Quick Test)")
    print("="*80)
    
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
        hidden_sizes=(128, 128),
        learning_rate=1e-3,
        batch_size=32,
        epochs_per_update=5,
    )
    
    print(f"\nTrainer Config:")
    print(f"  Device: {trainer.device}")
    print(f"  Policy: {sum(p.numel() for p in trainer.policy.parameters())} params")
    print(f"  Value: {sum(p.numel() for p in trainer.value.parameters())} params")
    
    # Quick training loop
    print(f"\nTraining for 3 episodes...")
    for episode in range(3):
        print(f"\n  Episode {episode+1}/3:")
        
        # Collect trajectory
        trajectory = trainer.collect_trajectory(max_steps=100)
        
        # Update
        losses = trainer.update(trajectory)
        
        print(f"    Policy Loss: {losses['policy_loss']:.4f}")
        print(f"    Value Loss: {losses['value_loss']:.4f}")
        
        if trainer.returns:
            print(f"    Episode Return: {trainer.returns[-1]:.2f}")
    
    env.close()
    
    print(f"\n✓ Example 3 Complete - Training works!\n")


def example_4_deterministic_control():
    """Example 4: Deterministic control trajectory."""
    print("="*80)
    print("Example 4: Deterministic Control Scenario")
    print("="*80)
    
    config = ToraxEnvironmentConfig(episode_max_steps=100)
    env = ToraxRLEnvironment(config)
    
    obs, _ = env.reset()
    print(f"\nScenario: Ramp current from 6 to 10 MA over episode")
    
    observations = [obs]
    actions_taken = []
    
    for step in range(50):
        # Linearly ramp current over first 30 steps
        if step < 30:
            target_dIp = 0.15  # Positive ramp
        else:
            target_dIp = 0.0   # Hold steady
        
        # 40% heating, ramping current
        action = np.array([0.4, target_dIp])
        obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs.copy())
        actions_taken.append(action)
        
        if step % 10 == 0:
            print(f"  Step {step}: Ip={obs[3]:.1f}MA, Te={obs[0]:.1f}keV, R={reward:.3f}")
        
        if terminated or truncated:
            break
    
    obs_array = np.array(observations)
    print(f"\nFinal State:")
    print(f"  Ip: {obs_array[0, 3]:.1f} → {obs_array[-1, 3]:.1f} MA")
    print(f"  Te: {obs_array[0, 0]:.1f} → {obs_array[-1, 0]:.1f} keV")
    print(f"  q95: {obs_array[0, 4]:.2f} → {obs_array[-1, 4]:.2f}")
    
    env.close()
    print("\n✓ Example 4 Complete\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("TORAX RL Environment - Complete Working Examples")
    print("="*80)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run examples
    try:
        example_1_basic_environment()
        example_2_qlknn_transport()
        example_4_deterministic_control()
        example_3_mini_training()
        
        print("="*80)
        print("✓✓✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("1. Train a full model: trainer.train(num_iterations=100)")
        print("2. Save and load checkpoints: trainer.save/load(path)")
        print("3. Evaluate on TORAX: set use_torax=True in config")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
