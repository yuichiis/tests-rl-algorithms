# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque 
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable
import gymnasium as gym

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#%matplotlib inline

    
# instantiate and test environment
cart_pole = gym.make("CartPole-v1")
n_actions_custom = cart_pole.action_space.n
n_observations_custom = cart_pole.observation_space.shape[0]

print(f"CartPole Environment:")
print(f"State Dim: {n_observations_custom}")
print(f"Action Dim: {n_actions_custom}")
start_state_tensor, _ = cart_pole.reset()
print(f"Example state tensor for (0,0): {start_state_tensor}")


# Define the Policy Network (Actor)
class PolicyNetwork(nn.Module):
    """ MLP Actor network for A2C """
    def __init__(self, n_observations: int, n_actions: int, units: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, n_actions)

    def forward(self, x: torch.Tensor) -> Categorical:
        """ Forward pass, returns a Categorical distribution. """
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        action_logits = self.layer3(x)
        return Categorical(logits=action_logits)

# Define the Value Network (Critic)
class ValueNetwork(nn.Module):
    """ MLP Critic network for A2C """
    def __init__(self, n_observations: int, units: int):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass, returns the estimated state value. """
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        state_value = self.layer3(x)
        return state_value
    

# Calculating Returns Advantages (using GAE)
def compute_gae_and_returns(rewards: torch.Tensor, 
                            values: torch.Tensor, 
                            next_values: torch.Tensor, 
                            dones: torch.Tensor, 
                            gamma: float, 
                            lambda_gae: float, 
                            standardize_adv: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and the returns-to-go (value targets).

    Parameters:
    - rewards, values, next_values, dones: Tensors collected from rollout.
    - gamma (float): Discount factor.
    - lambda_gae (float): GAE smoothing parameter.
    - standardize_adv (bool): Whether to standardize advantages.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: 
        - advantages: Tensor of GAE advantages.
        - returns_to_go: Tensor of target values for the critic (Advantage + V_old).
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    n_steps = len(rewards)

    # Calculate advantages using GAE formula backwards
    for t in reversed(range(n_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        advantages[t] = delta + gamma * lambda_gae * last_advantage * mask
        last_advantage = advantages[t]

    # Calculate returns-to-go (targets for value function)
    returns_to_go = advantages + values # R_t = A_t + V(s_t)

    # Standardize advantages (optional but usually recommended)
    if standardize_adv:
        mean_adv = torch.mean(advantages)
        std_adv = torch.std(advantages) + 1e-8
        advantages = (advantages - mean_adv) / std_adv
        
    return advantages, returns_to_go

# The A2C Update step
def update_a2c(actor: PolicyNetwork,
               critic: ValueNetwork,
               actor_optimizer: optim.Optimizer,
               critic_optimizer: optim.Optimizer, # Separate optimizers common
               states: torch.Tensor,
               actions: torch.Tensor,
               advantages: torch.Tensor,
               returns_to_go: torch.Tensor,
               value_loss_coeff: float,
               entropy_coeff: float) -> Tuple[float, float, float]:
    """
    Performs one synchronous update step for both Actor and Critic.

    Parameters:
    - actor, critic: The networks.
    - actor_optimizer, critic_optimizer: The optimizers.
    - states, actions, advantages, returns_to_go: Batch data tensors.
    - value_loss_coeff (float): Coefficient for the value loss.
    - entropy_coeff (float): Coefficient for the entropy bonus.

    Returns:
    - Tuple[float, float, float]: Policy loss, value loss, and entropy for this update step.
    """
    # --- Evaluate current networks --- 
    policy_dist = actor(states)
    log_probs = policy_dist.log_prob(actions)
    entropy = policy_dist.entropy().mean()
    values_pred = critic(states).squeeze()

    # --- Calculate Losses --- 
    # Policy Loss (Actor): - E[log_pi * A] - entropy_bonus
    # Advantages are detached as they shouldn't propagate gradients to the critic here
    policy_loss = -(log_probs * advantages.detach()).mean() - entropy_coeff * entropy
    
    # Value Loss (Critic): MSE(V_pred, R_target)
    # Returns-to-go are detached as they are targets
    value_loss = F.mse_loss(values_pred, returns_to_go.detach())
    
    # Combined loss (optional, can also optimize separately)
    # total_loss = policy_loss + value_loss_coeff * value_loss

    # --- Optimize Actor --- 
    actor_optimizer.zero_grad()
    policy_loss.backward() # Computes gradients only for actor parameters
    actor_optimizer.step()

    # --- Optimize Critic --- 
    critic_optimizer.zero_grad()
    # Need to scale value loss *before* backward if using combined loss and single backward()
    # If optimizing separately, backward() uses the unscaled loss.
    (value_loss_coeff * value_loss).backward() # Computes gradients only for critic parameters
    critic_optimizer.step()

    # Return individual loss components for logging
    return policy_loss.item() + entropy_coeff * entropy.item(), value_loss.item(), entropy.item()
    # Return the policy objective part (-log_pi*A), the value loss, and the entropy

# Hyperparameters for A2C on CartPole
GAMMA_A2C = 0.99             # Discount factor
GAE_LAMBDA_A2C = 0.95        # GAE lambda parameter
ACTOR_LR_A2C = 3e-4          # Learning rate for the actor
CRITIC_LR_A2C = 1e-3         # Learning rate for the critic
VALUE_LOSS_COEFF_A2C = 0.5   # Coefficient for value loss
ENTROPY_COEFF_A2C = 0.01     # Coefficient for entropy bonus
STANDARDIZE_ADV_A2C = True   # Whether to standardize advantages

NUM_ITERATIONS_A2C = 400    # Number of A2C updates (iterations)
STEPS_PER_ITERATION_A2C = 700#256 # Number of steps (batch size) collected per iteration
MAX_STEPS_PER_EPISODE_A2C = 600#200 # Max steps per episode
NETWORK_UNITS = 128 #256 #128


# Re-instantiate the environment
cart_pole = gym.make("CartPole-v1")
n_actions_custom: int = cart_pole.action_space.n
n_observations_custom: int = cart_pole.observation_space.shape[0]

# Initialize Actor and Critic
actor_a2c: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom, NETWORK_UNITS).to(device)
critic_a2c: ValueNetwork = ValueNetwork(n_observations_custom, NETWORK_UNITS).to(device)

# Initialize Optimizers
actor_optimizer_a2c: optim.Adam = optim.Adam(actor_a2c.parameters(), lr=ACTOR_LR_A2C)
critic_optimizer_a2c: optim.Adam = optim.Adam(critic_a2c.parameters(), lr=CRITIC_LR_A2C)

# Lists for plotting
a2c_iteration_rewards = []
a2c_iteration_avg_ep_lens = []
a2c_iteration_policy_losses = []
a2c_iteration_value_losses = []
a2c_iteration_entropies = []


# Training Loop
print("Starting A2C Training on CartPole...")

# --- A2C Training Loop ---
current_state, info = cart_pole.reset() # Start with initial state
current_state = torch.tensor(current_state, dtype=torch.float32, device=device)
#max_episode_length = 0
for iteration in range(NUM_ITERATIONS_A2C):
    # --- 1. Collect Trajectories (N steps) --- 
    batch_states_list: List[torch.Tensor] = []
    batch_actions_list: List[int] = []
    batch_log_probs_list: List[torch.Tensor] = []
    batch_rewards_list: List[float] = []
    batch_values_list: List[torch.Tensor] = [] # Store V(s_t)
    batch_dones_list: List[float] = []
    
    episode_rewards_in_iter: List[float] = []
    episode_lengths_in_iter: List[int] = []
    current_episode_reward = 0.0
    current_episode_length = 0

    for step in range(STEPS_PER_ITERATION_A2C):
        state_tensor = current_state
        
        # Sample action and get value estimate
        with torch.no_grad():
            policy_dist = actor_a2c(state_tensor)
            value_estimate = critic_a2c(state_tensor)
            action_tensor = policy_dist.sample()
            action = action_tensor.item()
            log_prob = policy_dist.log_prob(action_tensor)
            
        # Store data
        batch_states_list.append(state_tensor)
        batch_actions_list.append(action)
        batch_log_probs_list.append(log_prob)
        batch_values_list.append(value_estimate)

        # Interact
        next_state, reward, done, truncated, info = cart_pole.step(action)
        #if(truncated):
        #    print('reward',reward)
        #    print('done',done)
        #    print('truncated',truncated)
        #    print('info',info)
        #    print('current_episode_length',current_episode_length)

        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        batch_rewards_list.append(reward)
        batch_dones_list.append(float(done))
        
        current_state = next_state
        current_episode_reward += reward
        current_episode_length += 1
        
        # Handle episode termination within the batch collection
        if done or truncated or current_episode_length >= MAX_STEPS_PER_EPISODE_A2C:
            episode_rewards_in_iter.append(current_episode_reward)
            episode_lengths_in_iter.append(current_episode_length)
            current_state, info = cart_pole.reset() # Reset for the next episode
            current_state = torch.tensor(current_state, dtype=torch.float32, device=device)
            #if(max_episode_length<current_episode_length):
            #    max_episode_length = current_episode_length
            #    print('max_episode_length=',max_episode_length)
            current_episode_reward = 0.0
            current_episode_length = 0

    # --- End Rollout --- 

    # Bootstrap value for the last state if the episode didn't end
    with torch.no_grad():
        last_value = critic_a2c(current_state).squeeze() # Value of the state we'd start the next step from
        
    # Convert lists to tensors
    states_tensor = torch.stack(batch_states_list).to(device).squeeze(1) # Remove extra dim if added by network
    actions_tensor = torch.tensor(batch_actions_list, dtype=torch.long, device=device)
    log_probs_tensor = torch.stack(batch_log_probs_list).squeeze().to(device)
    rewards_tensor = torch.tensor(batch_rewards_list, dtype=torch.float32, device=device)
    values_tensor = torch.cat(batch_values_list).squeeze().to(device)
    dones_tensor = torch.tensor(batch_dones_list, dtype=torch.float32, device=device)

    # Need next_values for GAE calculation
    # Shift values and append the bootstrapped last_value
    next_values_tensor = torch.cat((values_tensor[1:], last_value.unsqueeze(0)))

    # --- 2. Estimate Advantages & Returns-to-go --- 
    advantages_tensor, returns_to_go_tensor = compute_gae_and_returns(
        rewards_tensor, values_tensor, next_values_tensor, dones_tensor, 
        GAMMA_A2C, GAE_LAMBDA_A2C, standardize_adv=STANDARDIZE_ADV_A2C
    )

    # --- 3. Perform A2C Update --- 
    avg_policy_loss, avg_value_loss, avg_entropy = update_a2c(
        actor_a2c, critic_a2c, actor_optimizer_a2c, critic_optimizer_a2c,
        states_tensor, actions_tensor, advantages_tensor, returns_to_go_tensor,
        VALUE_LOSS_COEFF_A2C, ENTROPY_COEFF_A2C
    )

    # --- Logging --- 
    avg_reward_iter = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
    avg_len_iter = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan

    a2c_iteration_rewards.append(avg_reward_iter)
    a2c_iteration_avg_ep_lens.append(avg_len_iter)
    a2c_iteration_policy_losses.append(avg_policy_loss)
    a2c_iteration_value_losses.append(avg_value_loss)
    a2c_iteration_entropies.append(avg_entropy)

    if (iteration + 1) % 50 == 0: # Print less frequently for potentially longer training
        print(f"Iter {iteration+1}/{NUM_ITERATIONS_A2C} | Avg Reward: {avg_reward_iter:.2f} | Avg Len: {avg_len_iter:.1f} | P_Loss: {avg_policy_loss:.4f} | V_Loss: {avg_value_loss:.4f} | Entropy: {avg_entropy:.4f}")

print("CartPole Training Finished (A2C).")

# Plotting results for A2C on CartPole
plt.figure(figsize=(20, 8))

# Average Rewards per Iteration
plt.subplot(2, 3, 1)
valid_rewards_a2c = [r for r in a2c_iteration_rewards if not np.isnan(r)]
valid_indices_a2c = [i for i, r in enumerate(a2c_iteration_rewards) if not np.isnan(r)]
plt.plot(valid_indices_a2c, valid_rewards_a2c)
plt.title('A2C CartPole: Avg Ep Reward / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg Reward')
plt.grid(True)
if len(valid_rewards_a2c) >= 20: # Use larger window for potentially noisy rewards
    rewards_ma_a2c = np.convolve(valid_rewards_a2c, np.ones(20)/20, mode='valid')
    plt.plot(valid_indices_a2c[19:], rewards_ma_a2c, label='20-iter MA', color='orange')
    plt.legend()

# Average Episode Length per Iteration
plt.subplot(2, 3, 2)
valid_lens_a2c = [l for l in a2c_iteration_avg_ep_lens if not np.isnan(l)]
valid_indices_len_a2c = [i for i, l in enumerate(a2c_iteration_avg_ep_lens) if not np.isnan(l)]
plt.plot(valid_indices_len_a2c, valid_lens_a2c)
plt.title('A2C CartPole: Avg Ep Length / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg Steps')
plt.grid(True)
if len(valid_lens_a2c) >= 20:
    lens_ma_a2c = np.convolve(valid_lens_a2c, np.ones(20)/20, mode='valid')
    plt.plot(valid_indices_len_a2c[19:], lens_ma_a2c, label='20-iter MA', color='orange')
    plt.legend()

# Critic (Value) Loss per Iteration
plt.subplot(2, 3, 3)
plt.plot(a2c_iteration_value_losses)
plt.title('A2C CartPole: Value Loss / Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.grid(True)
if len(a2c_iteration_value_losses) >= 20:
    vloss_ma_a2c = np.convolve(a2c_iteration_value_losses, np.ones(20)/20, mode='valid')
    plt.plot(np.arange(len(vloss_ma_a2c)) + 19, vloss_ma_a2c, label='20-iter MA', color='orange')
    plt.legend()

# Actor (Policy) Loss per Iteration (Policy Objective Part)
plt.subplot(2, 3, 4)
plt.plot(a2c_iteration_policy_losses)
plt.title('A2C CartPole: Policy Loss / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg (-log_pi*A - Ent)')
plt.grid(True)
if len(a2c_iteration_policy_losses) >= 20:
    ploss_ma_a2c = np.convolve(a2c_iteration_policy_losses, np.ones(20)/20, mode='valid')
    plt.plot(np.arange(len(ploss_ma_a2c)) + 19, ploss_ma_a2c, label='20-iter MA', color='orange')
    plt.legend()

# Entropy per Iteration
plt.subplot(2, 3, 5)
plt.plot(a2c_iteration_entropies)
plt.title('A2C CartPole: Policy Entropy / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.grid(True)
if len(a2c_iteration_entropies) >= 20:
    entropy_ma_a2c = np.convolve(a2c_iteration_entropies, np.ones(20)/20, mode='valid')
    plt.plot(np.arange(len(entropy_ma_a2c)) + 19, entropy_ma_a2c, label='20-iter MA', color='orange')
    plt.legend()

plt.tight_layout()
plt.show()