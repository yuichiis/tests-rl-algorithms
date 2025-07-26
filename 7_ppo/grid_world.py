# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque 
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable

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


# Custom Grid World Environment (Identical to the one in previous notebooks)
class GridEnvironment:
    """
    A simple 10x10 Grid World environment.
    State: (row, col) represented as normalized vector [row/9, col/9].
    Actions: 0 (up), 1 (down), 2 (left), 3 (right).
    Rewards: +10 for reaching the goal, -1 for hitting a wall, -0.1 for each step.
    """
    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
        self.state: Tuple[int, int] = self.start_state
        self.state_dim: int = 2
        self.action_dim: int = 4
        self.action_map: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self) -> torch.Tensor:
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
        norm_row = state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0
        norm_col = state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        normalized_state: List[float] = [norm_row, norm_col]
        return torch.tensor(normalized_state, dtype=torch.float32, device=device)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True
        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc
        reward: float = -0.1
        hit_wall: bool = False
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            next_row, next_col = current_row, current_col
            reward = -1.0
            hit_wall = True
        self.state = (next_row, next_col)
        next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)
        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0
        return next_state_tensor, reward, done

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim
    
custom_env = GridEnvironment(rows=10, cols=10)
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

print(f"Custom Grid Environment:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"State Dim: {n_observations_custom}")
print(f"Action Dim: {n_actions_custom}")
start_state_tensor = custom_env.reset()
print(f"Example state tensor for (0,0): {start_state_tensor}")

# Define the Policy Network (Actor)
class PolicyNetwork(nn.Module):
    """ MLP Actor network for PPO """
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> Categorical:
        """
        Forward pass, returns a Categorical distribution.
        """
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
    """ MLP Critic network for PPO """
    def __init__(self, n_observations: int):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, returns the estimated state value.
        """
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

# Define the compute_gae function that's currently missing
def compute_gae(rewards: torch.Tensor, 
               values: torch.Tensor, 
               next_values: torch.Tensor, 
               dones: torch.Tensor, 
               gamma: float, 
               lambda_gae: float, 
               standardize: bool = True) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        advantages[t] = delta + gamma * lambda_gae * last_advantage * mask
        last_advantage = advantages[t]

    if standardize:
        mean_adv = torch.mean(advantages)
        std_adv = torch.std(advantages) + 1e-8
        advantages = (advantages - mean_adv) / std_adv
        
    return advantages


def update_ppo(actor: PolicyNetwork,
               critic: ValueNetwork,
               actor_optimizer: optim.Optimizer,
               critic_optimizer: optim.Optimizer,
               states: torch.Tensor,
               actions: torch.Tensor,
               log_probs_old: torch.Tensor,
               advantages: torch.Tensor,
               returns_to_go: torch.Tensor,
               ppo_epochs: int,
               ppo_clip_epsilon: float,
               value_loss_coeff: float,
               entropy_coeff: float) -> Tuple[float, float, float]: # Return avg losses
    """
    Performs the PPO update for multiple epochs over the collected batch.

    Parameters:
    - actor, critic: The networks.
    - actor_optimizer, critic_optimizer: The optimizers.
    - states, actions, log_probs_old, advantages, returns_to_go: Batch data tensors.
    - ppo_epochs (int): Number of optimization epochs.
    - ppo_clip_epsilon (float): Clipping parameter epsilon.
    - value_loss_coeff (float): Coefficient for the value loss.
    - entropy_coeff (float): Coefficient for the entropy bonus.

    Returns:
    - Tuple[float, float, float]: Average policy loss, value loss, and entropy over the epochs.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    # Detach advantages and old log probs - they are treated as constants during the update
    advantages = advantages.detach()
    log_probs_old = log_probs_old.detach()
    returns_to_go = returns_to_go.detach()

    for _ in range(ppo_epochs):
        # --- Actor (Policy) Update --- 
        # Evaluate current policy
        policy_dist = actor(states)
        log_probs_new = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy().mean() # Entropy for exploration bonus
        
        # Calculate ratio r_t(theta)
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
        
        # PPO Clipped Policy Loss (negative because optimizer minimizes)
        # We add the entropy bonus (maximize entropy -> minimize negative entropy)
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy
        
        # Optimize the actor
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        
        # --- Critic (Value) Update --- 
        # Predict values
        values_pred = critic(states).squeeze()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values_pred, returns_to_go)
        
        # Optimize the critic
        critic_optimizer.zero_grad()
        # Scale value loss before backward pass
        (value_loss_coeff * value_loss).backward()
        critic_optimizer.step()
        
        # Accumulate losses for logging
        total_policy_loss += policy_loss.item() # Log negative clipped objective + entropy bonus
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        
    # Calculate average losses over the epochs
    avg_policy_loss = total_policy_loss / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    
    return avg_policy_loss, avg_value_loss, avg_entropy


# Hyperparameters for PPO on Custom Grid World
GAMMA_PPO = 0.99             # Discount factor
GAE_LAMBDA_PPO = 0.95        # GAE lambda parameter
PPO_CLIP_EPSILON = 0.2       # PPO clipping epsilon
ACTOR_LR = 3e-4              # Learning rate for the actor
CRITIC_LR_PPO = 1e-3         # Learning rate for the critic
PPO_EPOCHS = 10              # Number of optimization epochs per iteration
VALUE_LOSS_COEFF = 0.5       # Coefficient for value loss
ENTROPY_COEFF = 0.01         # Coefficient for entropy bonus
STANDARDIZE_ADV_PPO = True   # Whether to standardize advantages

NUM_ITERATIONS_PPO = 150     # Number of PPO iterations (policy updates)
STEPS_PER_ITERATION_PPO = 1000 # Steps collected per iteration
MAX_STEPS_PER_EPISODE_PPO = 200 # Max steps per episode


# Re-instantiate the environment
custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10)
n_actions_custom: int = custom_env.get_action_space_size()
n_observations_custom: int = custom_env.get_state_dimension()

# Initialize Actor and Critic
actor_ppo: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom).to(device)
critic_ppo: ValueNetwork = ValueNetwork(n_observations_custom).to(device)

# Initialize Optimizers
actor_optimizer_ppo: optim.Adam = optim.Adam(actor_ppo.parameters(), lr=ACTOR_LR)
critic_optimizer_ppo: optim.Adam = optim.Adam(critic_ppo.parameters(), lr=CRITIC_LR_PPO)

# Lists for plotting
ppo_iteration_rewards = []
ppo_iteration_avg_ep_lens = []
ppo_iteration_policy_losses = []
ppo_iteration_value_losses = []
ppo_iteration_entropies = []


print("Starting PPO Training on Custom Grid World...")

# --- PPO Training Loop ---
for iteration in range(NUM_ITERATIONS_PPO):
    # --- 1. Collect Trajectories (Rollout Phase) --- 
    # Store data in lists temporarily
    batch_states_list = []
    batch_actions_list = []
    batch_log_probs_old_list = []
    batch_rewards_list = []
    batch_values_list = []
    batch_dones_list = []
    
    episode_rewards_in_iter = []
    episode_lengths_in_iter = []
    steps_collected = 0

    while steps_collected < STEPS_PER_ITERATION_PPO:
        state = custom_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        for t in range(MAX_STEPS_PER_EPISODE_PPO):
            # Sample action and get value estimate
            with torch.no_grad():
                policy_dist = actor_ppo(state)
                value = critic_ppo(state).squeeze()
                
                action_tensor = policy_dist.sample()
                action = action_tensor.item()
                log_prob = policy_dist.log_prob(action_tensor)
            
            # Interact with environment
            next_state, reward, done = custom_env.step(action)
            
            # Store data
            batch_states_list.append(state)
            batch_actions_list.append(action)
            batch_log_probs_old_list.append(log_prob)
            batch_values_list.append(value)
            batch_rewards_list.append(reward)
            batch_dones_list.append(float(done))
            
            state = next_state
            episode_reward += reward
            steps_collected += 1
            episode_steps += 1
            
            if done or steps_collected >= STEPS_PER_ITERATION_PPO:
                episode_rewards_in_iter.append(episode_reward)
                episode_lengths_in_iter.append(episode_steps)
                break
                
            if steps_collected >= STEPS_PER_ITERATION_PPO:
                break
    
    # --- End Rollout --- 

    # Calculate next_values for GAE
    # For non-terminal states, next_value is the value of the next state
    # For terminal states, next_value is 0
    next_values = []
    with torch.no_grad():
        for i in range(len(batch_states_list)):
            if batch_dones_list[i] > 0.5:  # If done
                next_values.append(torch.tensor(0.0))
            elif i == len(batch_states_list) - 1:  # Last state in batch
                next_state = custom_env.step(batch_actions_list[i])[0]  # Get next state
                next_values.append(critic_ppo(next_state).squeeze())
            else:  # Not done, use value of next state in batch
                next_values.append(batch_values_list[i+1])

    # Convert lists to tensors
    states_tensor = torch.stack(batch_states_list).to(device)
    actions_tensor = torch.tensor(batch_actions_list, dtype=torch.long, device=device)
    log_probs_old_tensor = torch.stack(batch_log_probs_old_list).squeeze().to(device)
    rewards_tensor = torch.tensor(batch_rewards_list, dtype=torch.float32, device=device)
    values_tensor = torch.stack(batch_values_list).to(device)
    next_values_tensor = torch.stack(next_values).to(device)
    dones_tensor = torch.tensor(batch_dones_list, dtype=torch.float32, device=device)

    # --- 2. Estimate Advantages & Returns-to-go --- 
    advantages_tensor = compute_gae(
        rewards_tensor, values_tensor, next_values_tensor, dones_tensor, 
        GAMMA_PPO, GAE_LAMBDA_PPO, standardize=STANDARDIZE_ADV_PPO
    )
    returns_to_go_tensor = advantages_tensor + values_tensor

    # --- 3. Perform PPO Update --- 
    avg_policy_loss, avg_value_loss, avg_entropy = update_ppo(
        actor_ppo, critic_ppo, actor_optimizer_ppo, critic_optimizer_ppo,
        states_tensor, actions_tensor, log_probs_old_tensor,
        advantages_tensor, returns_to_go_tensor,
        PPO_EPOCHS, PPO_CLIP_EPSILON, VALUE_LOSS_COEFF, ENTROPY_COEFF
    )

    # --- Logging --- 
    avg_reward_iter = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
    avg_len_iter = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan

    ppo_iteration_rewards.append(avg_reward_iter)
    ppo_iteration_avg_ep_lens.append(avg_len_iter)
    ppo_iteration_policy_losses.append(avg_policy_loss)
    ppo_iteration_value_losses.append(avg_value_loss)
    ppo_iteration_entropies.append(avg_entropy)

    if (iteration + 1) % 10 == 0:
        print(f"Iter {iteration+1}/{NUM_ITERATIONS_PPO} | Avg Reward: {avg_reward_iter:.2f} | Avg Len: {avg_len_iter:.1f} | P_Loss: {avg_policy_loss:.4f} | V_Loss: {avg_value_loss:.4f} | Entropy: {avg_entropy:.4f}")

print("Custom Grid World Training Finished (PPO).")



# Plotting results for PPO on Custom Grid World
plt.figure(figsize=(20, 8))

# Average Rewards per Iteration
plt.subplot(2, 3, 1)
valid_rewards_ppo = [r for r in ppo_iteration_rewards if not np.isnan(r)]
valid_indices_ppo = [i for i, r in enumerate(ppo_iteration_rewards) if not np.isnan(r)]
plt.plot(valid_indices_ppo, valid_rewards_ppo)
plt.title('PPO Custom Grid: Avg Ep Reward / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg Reward')
plt.grid(True)
if len(valid_rewards_ppo) >= 10:
    rewards_ma_ppo = np.convolve(valid_rewards_ppo, np.ones(10)/10, mode='valid')
    plt.plot(valid_indices_ppo[9:], rewards_ma_ppo, label='10-iter MA', color='orange')
    plt.legend()

# Average Episode Length per Iteration
plt.subplot(2, 3, 2)
valid_lens_ppo = [l for l in ppo_iteration_avg_ep_lens if not np.isnan(l)]
valid_indices_len_ppo = [i for i, l in enumerate(ppo_iteration_avg_ep_lens) if not np.isnan(l)]
plt.plot(valid_indices_len_ppo, valid_lens_ppo)
plt.title('PPO Custom Grid: Avg Ep Length / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg Steps')
plt.grid(True)
if len(valid_lens_ppo) >= 10:
    lens_ma_ppo = np.convolve(valid_lens_ppo, np.ones(10)/10, mode='valid')
    plt.plot(valid_indices_len_ppo[9:], lens_ma_ppo, label='10-iter MA', color='orange')
    plt.legend()

# Critic (Value) Loss per Iteration
plt.subplot(2, 3, 3)
plt.plot(ppo_iteration_value_losses)
plt.title('PPO Custom Grid: Avg Value Loss / Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.grid(True)
if len(ppo_iteration_value_losses) >= 10:
    vloss_ma_ppo = np.convolve(ppo_iteration_value_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(vloss_ma_ppo)) + 9, vloss_ma_ppo, label='10-iter MA', color='orange')
    plt.legend()

# Actor (Policy) Loss per Iteration
plt.subplot(2, 3, 4)
# Plotting the negative loss (since we minimized -L_clip - entropy)
plt.plot([-l for l in ppo_iteration_policy_losses]) 
plt.title('PPO Custom Grid: Avg Policy Objective / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Avg (-Policy Loss)') 
plt.grid(True)
if len(ppo_iteration_policy_losses) >= 10:
    ploss_ma_ppo = np.convolve([-l for l in ppo_iteration_policy_losses], np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(ploss_ma_ppo)) + 9, ploss_ma_ppo, label='10-iter MA', color='orange')
    plt.legend()

# Entropy per Iteration
plt.subplot(2, 3, 5)
plt.plot(ppo_iteration_entropies)
plt.title('PPO Custom Grid: Avg Policy Entropy / Iteration')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.grid(True)
if len(ppo_iteration_entropies) >= 10:
    entropy_ma_ppo = np.convolve(ppo_iteration_entropies, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(entropy_ma_ppo)) + 9, entropy_ma_ppo, label='10-iter MA', color='orange')
    plt.legend()

plt.tight_layout()
plt.show()



# Reusing the policy plotting function (works for any network outputting Categorical)
def plot_ppo_policy_grid(policy_net: PolicyNetwork, env: GridEnvironment, device: torch.device) -> None:
    """
    Plots the greedy policy derived from the PPO policy network.
    Shows the most likely action for each state.
    (Identical to the REINFORCE/TRPO plotting function)
    """
    rows: int = env.rows
    cols: int = env.cols
    policy_grid: np.ndarray = np.empty((rows, cols), dtype=str)
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.6))

    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)
            if state_tuple == env.goal_state:
                policy_grid[r, c] = 'G'
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=12, weight='bold')
            else:
                state_tensor: torch.Tensor = env._get_state_tensor(state_tuple)
                with torch.no_grad():
                    action_dist: Categorical = policy_net(state_tensor)
                    best_action: int = action_dist.probs.argmax(dim=1).item()

                policy_grid[r, c] = action_symbols[best_action]
                ax.text(c, r, policy_grid[r, c], ha='center', va='center', color='black', fontsize=12)

    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("PPO Learned Policy (Most Likely Action)")
    plt.show()

# Plot the policy learned by the trained PPO actor
print("\nPlotting Learned Policy from PPO:")
plot_ppo_policy_grid(actor_ppo, custom_env, device)



