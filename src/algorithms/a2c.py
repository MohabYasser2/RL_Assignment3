"""
A2C (Advantage Actor-Critic) Algorithm Implementation.

This module contains the A2C algorithm for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, Any, Tuple, List
import numpy as np
import gymnasium as gym
class PendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # logits for discrete actions
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)


class PendulumCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for A2C.
    Shares feature extraction layers between actor and critic.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, is_continuous: bool = False):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
            is_continuous: Whether action space is continuous
        """
        super(ActorCritic, self).__init__()
        
        self.is_continuous = is_continuous
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        if is_continuous:
            # For continuous actions, output mean and log_std
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            # Initialize log_std to -0.5 for initial std of exp(-0.5) ≈ 0.6
            self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        else:
            # For discrete actions, output action probabilities
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            
        Returns:
            action_probs/mean, value
        """
        features = self.shared_layers(state)
        
        if self.is_continuous:
            action_mean = self.actor_mean(features)
            # Clamp log_std to prevent numerical instability
            action_log_std = torch.clamp(self.actor_log_std, -20, 2)
            action_std = torch.exp(action_log_std)
            value = self.critic(features)
            return action_mean, action_std, value
        else:
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """
        Get action distribution, sample action, and compute value.
        
        Args:
            state: Input state
            action: Optional action for computing log probability
            
        Returns:
            action, log_prob, entropy, value
        """
        if self.is_continuous:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)
            
            if action is None:
                action = dist.sample()
                # Clip action to reasonable range (will be further clipped by environment)
                action = torch.clamp(action, -3, 3)
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
        else:
            action_logits, value = self.forward(state)
            dist = Categorical(logits=action_logits)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class A2C:
    """
    Advantage Actor-Critic (A2C) algorithm.
    
    A2C is a synchronous variant of A3C that combines policy gradient methods
    with value function approximation to reduce variance.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize the A2C agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary with hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if action space is continuous (supports both 'continuous' and 'is_continuous')
        self.is_continuous = config.get('continuous', config.get('is_continuous', False))
        
        # Hyperparameters
        self.gamma = config.get('discount_factor', 0.99)
        self.lr = config.get('learning_rate', 0.0007)
        self.n_steps = config.get('n_steps', 5)  # n-step returns
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gae_lambda = config.get('gae_lambda', 0.95)  # GAE lambda parameter
        
        # Initialize actor-critic network
        hidden_dim = config.get('hidden_dim', 128)
        
        # Check if we should use separate networks (for Pendulum)
        self.use_separate_networks = action_dim == 5 and state_dim == 3  # Pendulum detection
        
        if self.use_separate_networks:
            # Use separate actor and critic networks for Pendulum
            self.actor = PendulumActor(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic = PendulumCritic(state_dim, hidden_dim).to(self.device)
            
            # Separate optimizers with different learning rates
            actor_lr = config.get('actor_lr', self.lr)
            critic_lr = config.get('critic_lr', self.lr * 2.5)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            # Use shared network for other environments
            self.actor_critic = ActorCritic(
                state_dim, 
                action_dim, 
                hidden_dim, 
                is_continuous=self.is_continuous
            ).to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.actor_critic.parameters(), 
                lr=self.lr
            )
        
        # Storage for trajectories
        self.reset_storage()
        
    def reset_storage(self):
        """Reset trajectory storage."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []
        
    def train(self, env, config: Dict = None, logger=None, num_episodes: int = None) -> Dict[str, Any]:
        """
        Train the A2C agent.
        
        Args:
            env: Environment to train on
            config: Configuration dictionary (optional, uses self config if not provided)
            logger: Logger instance for tracking metrics (optional)
            num_episodes: Number of episodes to train for (optional, uses config if not provided)
            
        Returns:
            Dictionary containing training metrics and statistics
        """
        # Use provided config or fall back to instance config
        if config is None:
            config = {}
        
        # Determine convergence settings or fixed episodes
        use_convergence = 'convergence_threshold' in config
        if use_convergence:
            convergence_threshold = config['convergence_threshold']
            convergence_window = config.get('convergence_window', 100)
            min_episodes = config.get('min_episodes', 100)
            max_episodes = config.get('max_episodes', 5000)
            num_episodes = max_episodes  # Use max as loop limit
        else:
            # Determine number of episodes
            if num_episodes is None:
                num_episodes = config.get('episodes', 1000)
        
        log_interval = config.get('log_interval', 10)
        
        episode_rewards = []
        episode_lengths = []
        all_losses = []
        recent_rewards = []  # For computing rolling average
        best_avg_reward = -float('inf')  # Track best performance
        best_model_state = None  # Store best model
        
        from tqdm import tqdm
        
        for episode in tqdm(range(num_episodes), desc="Training A2C"):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            self.reset_storage()
            
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Select action using current policy
                with torch.no_grad():
                    if self.use_separate_networks:
                        # Separate networks
                        logits = self.actor(state_tensor)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        entropy = dist.entropy()
                        value = self.critic(state_tensor)
                    else:
                        # Shared network
                        action, log_prob, entropy, value = self.actor_critic.get_action_and_value(state_tensor)
                
                # Execute action in environment
                if self.is_continuous:
                    next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                else:
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                done = terminated or truncated
                
                # Store transition
                self.states.append(state_tensor)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                self.values.append(value)
                self.dones.append(done)
                self.entropies.append(entropy)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                # Update every n_steps or at episode end
                if len(self.rewards) >= self.n_steps or done:
                    # Get value of next state for bootstrapping
                    if done:
                        next_value = 0.0
                    else:
                        with torch.no_grad():
                            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                            if self.use_separate_networks:
                                next_value = self.critic(next_state_tensor).item()
                            elif self.is_continuous:
                                _, _, next_value = self.actor_critic(next_state_tensor)
                                next_value = next_value.item()
                            else:
                                _, next_value = self.actor_critic(next_state_tensor)
                                next_value = next_value.item()
                    
                    # Compute returns and advantages
                    returns, advantages = self.compute_returns_and_advantages(
                        next_value, self.dones, self.rewards, self.values
                    )
                    
                    # Prepare batch
                    states_batch = torch.cat(self.states, dim=0)
                    actions_batch = torch.stack(self.actions).view(-1)

                    
                    # Compute loss
                    if self.use_separate_networks:
                        loss_dict = self.compute_loss_separate(
                            states_batch, actions_batch, returns, advantages
                        )
                    else:
                        loss, loss_dict = self.compute_loss(
                            states_batch, actions_batch, returns, advantages
                        )
                        
                        # Optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_critic.parameters(), 
                            self.max_grad_norm
                        )
                        
                        self.optimizer.step()
                    
                    all_losses.append(loss_dict)
                    
                    # Reset storage for next batch
                    self.reset_storage()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            recent_rewards.append(episode_reward)
            
            # Keep only recent rewards for rolling average
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            # Log to W&B
            if logger and (episode + 1) % log_interval == 0:
                avg_reward_recent = np.mean(recent_rewards)
                log_data = {
                    'episode': episode + 1,
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'avg_reward_100': avg_reward_recent,
                }
                
                # Add loss metrics if available
                if all_losses:
                    last_loss = all_losses[-1]
                    log_data.update(last_loss)
                
                logger.log(log_data)
            
            # Print progress
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
                avg_length = np.mean(episode_lengths[-min(100, len(episode_lengths)):])
                
                # Track best model based on 100-episode average
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    # Save a copy of the best model state
                    if self.use_separate_networks:
                        best_model_state = {
                            'actor_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                            'episode': episode + 1,
                            'avg_reward': avg_reward
                        }
                    else:
                        best_model_state = {
                            'actor_critic_state_dict': self.actor_critic.state_dict().copy(),
                            'optimizer_state_dict': self.optimizer.state_dict().copy(),
                            'episode': episode + 1,
                            'avg_reward': avg_reward
                        }
                    tqdm.write(f"🌟 New best avg reward: {best_avg_reward:.2f} at episode {episode + 1}")
                
                tqdm.write(f"Episode {episode + 1}/{num_episodes} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Avg Reward (100): {avg_reward:.2f} | "
                          f"Avg Length: {avg_length:.2f}")
            
            # Check for convergence
            if use_convergence and episode + 1 >= min_episodes:
                # Use the appropriate window size
                window_size = min(convergence_window, len(episode_rewards))
                avg_reward_window = np.mean(episode_rewards[-window_size:])
                
                if avg_reward_window >= convergence_threshold:
                    tqdm.write(f"\n🎉 Converged! Average reward over last {window_size} episodes: "
                              f"{avg_reward_window:.2f} >= {convergence_threshold:.2f}")
                    tqdm.write(f"Training stopped at episode {episode + 1}")
                    break
        
        # Restore best model before returning
        if best_model_state is not None:
            if self.use_separate_networks:
                self.actor.load_state_dict(best_model_state['actor_state_dict'])
                self.critic.load_state_dict(best_model_state['critic_state_dict'])
                self.actor_optimizer.load_state_dict(best_model_state['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(best_model_state['critic_optimizer_state_dict'])
            else:
                self.actor_critic.load_state_dict(best_model_state['actor_critic_state_dict'])
                self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            print(f"\n✅ Restored best model from episode {best_model_state['episode']} "
                  f"with avg reward {best_model_state['avg_reward']:.2f}")
        
        # Final statistics
        final_stats = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'losses': all_losses,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'final_avg_reward_100': np.mean(episode_rewards[-min(100, len(episode_rewards)):])
        }
        
        return final_stats
    
    def test(self, env, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Test the A2C agent.
        
        Args:
            env: Environment to test on
            num_episodes: Number of test episodes
            
        Returns:
            Dictionary containing test metrics and statistics
        """
        # Set to evaluation mode
        if self.use_separate_networks:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor_critic.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action deterministically
                action = self.select_action(state, deterministic=True)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Set back to training mode
        if self.use_separate_networks:
            self.actor.train()
            self.critic.train()
        else:
            self.actor_critic.train()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Any:
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            deterministic: If True, select action deterministically
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_separate_networks:
                # Separate networks (Pendulum)
                logits = self.actor(state_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                return action.item()
            elif self.is_continuous:
                action_mean, action_std, _ = self.actor_critic(state_tensor)
                
                if deterministic:
                    action = action_mean
                else:
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    # Clip action to reasonable range
                    action = torch.clamp(action, -3, 3)
                
                return action.cpu().numpy()[0]
            else:
                action_logits, _ = self.actor_critic(state_tensor)
                
                if deterministic:
                    action = torch.argmax(action_logits, dim=-1)
                else:
                    dist = Categorical(logits=action_logits)
                    action = dist.sample()
                
                return action.item()
    
    def compute_returns_and_advantages(self, next_value: float, dones: List[bool], 
                                       rewards: List[float], values: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute n-step returns and advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            next_value: Value estimate for the next state
            dones: List of done flags
            rewards: List of rewards
            values: List of value estimates
            
        Returns:
            returns, advantages
        """
        returns = []
        advantages = []
        
        # Compute n-step returns and advantages
        gae = 0
        next_value_tensor = torch.tensor(next_value).to(self.device)
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value_step = next_value_tensor
            else:
                next_value_step = values[step + 1]
            
            # TD error: δ = r + γ * V(s') - V(s)
            delta = rewards[step] + self.gamma * next_value_step * (1 - dones[step]) - values[step]
            
            # GAE: A = δ + γλ * A
            # λ controls bias-variance tradeoff: λ=0 (high bias), λ=1 (high variance)
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        returns = torch.stack(returns).squeeze(-1)
        advantages = torch.stack(advantages).squeeze(-1)
        return returns, advantages

    def compute_loss_separate(self, states: torch.Tensor, actions: torch.Tensor, 
                    returns: torch.Tensor, advantages: torch.Tensor) -> Dict:
        """
        Compute loss for separate actor-critic networks (Pendulum).
        Updates both networks separately with their own optimizers.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            returns: Batch of computed returns
            advantages: Batch of computed advantages
            
        Returns:
            loss_dict with loss values
        """
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Critic update
        values = self.critic(states)
        value_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Actor update
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        loss_dict = {
            'total_loss': (actor_loss + self.value_loss_coef * value_loss).item(),
            'policy_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
        
        return loss_dict
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                    returns: torch.Tensor, advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the A2C loss function.
        
        Loss = Policy Loss + Value Loss - Entropy Bonus
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            returns: Batch of computed returns
            advantages: Batch of computed advantages
            
        Returns:
            total_loss, loss_dict
        """
        # Get action log probabilities, entropy, and values from current policy
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        _, log_probs, entropy, values = self.actor_critic.get_action_and_value(states, actions)
        
        # Normalize advantages (helps with training stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        # Policy loss: -log π(a|s) * A(s,a)
        # We want to maximize this, so we minimize the negative
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss: MSE between predicted values and returns
        # Ensure both tensors have same shape
        returns_flat = returns.view(-1)
        values_flat = values.view(-1)
        value_loss = F.mse_loss(values_flat, returns_flat)
        
        # Entropy bonus: encourages exploration
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
        
        return total_loss, loss_dict
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        if self.use_separate_networks:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'config': self.config
            }, path)
        else:
            torch.save({
                'actor_critic_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        if self.use_separate_networks:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        else:
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
