"""
PPO (Proximal Policy Optimization) Trainer for TORAX RL Environment

Implements full PPO algorithm with:
- Actor (policy) network learning action distribution
- Critic (value) network learning state value function
- Generalized Advantage Estimation (GAE)
- Clipped surrogate loss with entropy regularization
- Batch training with multiple epochs
- Checkpointing and logging

Reference: Schulman et al. 2017 (PPO paper)
Author: Research Implementation
Date: March 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Actor network for policy gradient methods.
    
    Maps observations to action mean and log std.
    Uses separate output heads for continuous action space.
    """
    
    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 2,
        hidden_sizes: Tuple[int, int] = (256, 256),
        activation: nn.Module = nn.Tanh,
    ):
        """Initialize policy network."""
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared trunk
        layers = []
        in_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation())
            in_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Linear(in_size, action_dim)
        
        # Log std head (learnable parameter, not output)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns action mean and std.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            
        Returns:
            mean: Action mean of shape (batch, action_dim)
            std: Action std of shape (batch, action_dim)
        """
        x = self.trunk(obs)
        mean = self.mean_head(x)
        
        # Ensure std is positive and proper shape
        std = torch.exp(self.log_std).unsqueeze(0).expand(mean.shape[0], -1)
        
        return mean, std
    
    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            obs: Observation tensor
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability for given action.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            log_prob: Log probability
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob


class ValueNetwork(nn.Module):
    """Critic network for value function approximation.
    
    Maps observations to state value estimates.
    Used for baseline in advantage computation.
    """
    
    def __init__(
        self,
        obs_dim: int = 11,
        hidden_sizes: Tuple[int, int] = (256, 256),
        activation: nn.Module = nn.Tanh,
    ):
        """Initialize value network."""
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # Network
        layers = []
        in_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returns value estimate.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            
        Returns:
            value: Value estimate of shape (batch, 1)
        """
        return self.network(obs)


class PPOTrainer:
    """PPO Algorithm Implementation.
    
    Trains policy and value networks using PPO updates.
    Collects experience trajectories and performs mini-batch SGD.
    """
    
    def __init__(
        self,
        env,
        obs_dim: int = 11,
        action_dim: int = 2,
        hidden_sizes: Tuple[int, int] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.0,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        epochs_per_update: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize PPO trainer.
        
        Args:
            env: Gymnasium environment
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss weight
            max_grad_norm: Gradient clipping threshold
            batch_size: Mini-batch size for updates
            epochs_per_update: Epochs per update phase
            device: Device for computation (cuda/cpu)
        """
        self.env = env
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        
        # Networks
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
        # Tracking
        self.total_steps = 0
        self.episode_count = 0
        self.returns = []
    
    def collect_trajectory(self, max_steps: int = 1000) -> Dict:
        """Collect experience trajectory from environment.
        
        Args:
            max_steps: Maximum steps to collect
            
        Returns:
            Dictionary with trajectory data
        """
        obs_list = []
        action_list = []
        reward_list = []
        value_list = []
        log_prob_list = []
        done_list = []
        
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        episode_return = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.policy.sample_action(obs.unsqueeze(0))
                value = self.value(obs.unsqueeze(0))
            
            action = action.squeeze(0).cpu().numpy()
            log_prob = log_prob.squeeze(0).item()
            value = value.squeeze(0).item()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            obs_list.append(obs.cpu().numpy())
            action_list.append(action)
            reward_list.append(reward)
            value_list.append(value)
            log_prob_list.append(log_prob)
            done_list.append(done)
            
            # Update
            episode_return += reward
            episode_steps += 1
            self.total_steps += 1
            
            if done:
                obs, _ = self.env.reset()
                self.episode_count += 1
                self.returns.append(episode_return)
                
                if len(self.returns) % 10 == 0:
                    avg_return = np.mean(self.returns[-10:])
                    logger.info(f"Episode {self.episode_count}: Return={episode_return:.2f}, Avg={avg_return:.2f}")
                
                episode_return = 0.0
                episode_steps = 0
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        
        # Compute advantages with GAE
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            next_values = self.value(obs_tensor).squeeze(-1).cpu().numpy()
        
        advantages = self._compute_gae(
            np.array(reward_list),
            np.array(value_list),
            next_values,
            np.array(done_list),
        )
        
        returns = advantages + np.array(value_list)
        
        trajectory = {
            "obs": obs_list,
            "action": action_list,
            "reward": reward_list,
            "value": value_list,
            "log_prob": log_prob_list,
            "advantage": advantages,
            "return": returns,
        }
        
        return trajectory
    
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            next_values: Next state value estimates
            dones: Done flags
            
        Returns:
            Computed advantages
        """
        advantages = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def update(self, trajectory: Dict) -> Dict[str, float]:
        """Perform PPO update.
        
        Args:
            trajectory: Trajectory dictionary from collect_trajectory
            
        Returns:
            Dictionary with loss values
        """
        # Convert to tensors
        obs = torch.tensor(np.array(trajectory["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(trajectory["action"]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(trajectory["log_prob"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(trajectory["advantage"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(trajectory["return"], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform multiple epochs of update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        num_batches = max(1, len(obs) // self.batch_size)
        
        for epoch in range(self.epochs_per_update):
            # Shuffle indices
            indices = np.random.permutation(len(obs))
            
            for batch_idx in range(num_batches):
                # Get batch
                batch_indices = indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Policy update
                new_log_probs = self.policy.get_log_prob(batch_obs, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy regularization
                if self.entropy_coef > 0:
                    entropy = -(new_log_probs.exp() * new_log_probs).mean()
                    policy_loss = policy_loss - self.entropy_coef * entropy
                
                # Value update
                values = self.value(batch_obs).squeeze(-1)
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Policy optimization step
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Value optimization step
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        num_updates = num_batches * self.epochs_per_update
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
        }
    
    def train(
        self,
        num_iterations: int = 100,
        steps_per_iteration: int = 2048,
        save_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            steps_per_iteration: Steps to collect per iteration
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        history = {
            "policy_loss": [],
            "value_loss": [],
            "episode_return": [],
        }
        
        logger.info(f"Starting PPO training: {num_iterations} iterations, {steps_per_iteration} steps/iter")
        
        for iteration in range(num_iterations):
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps=steps_per_iteration)
            
            # Update networks
            losses = self.update(trajectory)
            
            history["policy_loss"].append(losses["policy_loss"])
            history["value_loss"].append(losses["value_loss"])
            
            if self.returns:
                history["episode_return"].append(self.returns[-1])
            
            if (iteration + 1) % 10 == 0:
                avg_return = np.mean(self.returns[-10:]) if len(self.returns) >= 10 else np.mean(self.returns)
                logger.info(
                    f"Iter {iteration+1}/{num_iterations}: "
                    f"PolicyLoss={losses['policy_loss']:.4f}, "
                    f"ValueLoss={losses['value_loss']:.4f}, "
                    f"AvgReturn={avg_return:.2f}"
                )
            
            # Save checkpoint
            if save_dir and (iteration + 1) % 50 == 0:
                self.save(os.path.join(save_dir, f"checkpoint_iter{iteration+1}.pt"))
        
        return history
    
    def save(self, path: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.episode_count = checkpoint["episode_count"]
        logger.info(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    # Quick test
    print("\nPPO Trainer Module Loaded Successfully")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  PyTorch: {torch.__version__}\n")
