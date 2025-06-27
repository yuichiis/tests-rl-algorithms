# Import necessary libraries for numerical computations, plotting, and utility functions
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque # Deque might not be needed for REINFORCE
from itertools import count
from typing import List, Tuple, Dict, Optional

# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical # Needed for sampling actions

# Set up the device to use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a random seed for reproducibility across runs
seed = 42
random.seed(seed)  # Seed for Python's random module
np.random.seed(seed)  # Seed for NumPy
torch.manual_seed(seed)  # Seed for PyTorch (CPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # Seed for PyTorch (GPU)

# Enable inline plotting for Jupyter Notebook
#%matplotlib inline

# Custom Grid World Environment (Identical to the one in DQN notebook)
class GridEnvironment:
    """
    A simple 10x10 Grid World environment.
    State: (row, col) represented as normalized vector [row/10, col/10].
    Actions: 0 (up), 1 (down), 2 (left), 3 (right).
    Rewards: +10 for reaching the goal, -1 for hitting a wall, -0.1 for each step.
    """

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        """
        Initializes the Grid World environment.

        Parameters:
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.
        """
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)  # Starting position
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)  # Goal position
        self.state: Tuple[int, int] = self.start_state  # Current state
        self.state_dim: int = 2  # State represented by 2 coordinates (row, col)
        self.action_dim: int = 4  # 4 discrete actions: up, down, left, right

        # Action mapping: maps action index to (row_delta, col_delta)
        self.action_map: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self) -> torch.Tensor:
        """
        Resets the environment to the start state.

        Returns:
            torch.Tensor: The initial state as a normalized tensor.
        """
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
        """
        Converts a (row, col) tuple to a normalized tensor for the network.

        Parameters:
        - state_tuple (Tuple[int, int]): The state represented as a tuple (row, col).

        Returns:
            torch.Tensor: The normalized state as a tensor.
        """
        # Normalize coordinates to be between 0 and 1 (adjust normalization slightly for 0-based indexing)
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0,
            state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        ]
        return torch.tensor(normalized_state, dtype=torch.float32, device=device)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Performs one step in the environment based on the given action.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            Tuple[torch.Tensor, float, bool]: 
                - next_state_tensor (torch.Tensor): The next state as a normalized tensor.
                - reward (float): The reward for the action.
                - done (bool): Whether the episode has ended.
        """
        # If the goal state is already reached, return the current state with 0 reward and done=True
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True

        # Get the row and column deltas for the action
        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        # Default step cost
        reward: float = -0.1
        hit_wall: bool = False

        # Check if the action leads to hitting a wall (out of bounds)
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            # Stay in the same state and incur a penalty
            next_row, next_col = current_row, current_col
            reward = -1.0
            hit_wall = True

        # Update the state
        self.state = (next_row, next_col)
        next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)

        # Check if the goal state is reached
        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0  # Reward for reaching the goal

        return next_state_tensor, reward, done

    def get_action_space_size(self) -> int:
        """
        Returns the size of the action space.

        Returns:
            int: The number of possible actions (4).
        """
        return self.action_dim

    def get_state_dimension(self) -> int:
        """
        Returns the dimension of the state representation.

        Returns:
            int: The number of dimensions in the state (2).
        """
        return self.state_dim
    
# Instantiate the custom grid environment with a 10x10 grid
custom_env = GridEnvironment(rows=10, cols=10)

# Get the size of the action space and state dimension
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

# Print basic information about the environment
print(f"Custom Grid Environment:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"State Dim: {n_observations_custom}")
print(f"Action Dim: {n_actions_custom}")
print(f"Start State: {custom_env.start_state}")
print(f"Goal State: {custom_env.goal_state}")

# Reset the environment and print the normalized state tensor for the start state
print(f"Example state tensor for (0,0): {custom_env.reset()}")

# Take an example step: move 'right' (action=3) and print the result
next_s, r, d = custom_env.step(3) # Action 3 corresponds to moving right
print(f"Step result (action=right): next_state={next_s.cpu().numpy()}, reward={r}, done={d}")

# Take another example step: move 'up' (action=0) and print the result
# This should hit a wall since the agent is at the top row
next_s, r, d = custom_env.step(0) # Action 0 corresponds to moving up
print(f"Step result (action=up): next_state={next_s.cpu().numpy()}, reward={r}, done={d}")


# Define the Policy Network architecture
class PolicyNetwork(nn.Module):
    """ Simple MLP Policy Network for REINFORCE """
    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize the Policy Network.

        Parameters:
        - n_observations (int): Dimension of the state space.
        - n_actions (int): Number of possible actions.
        """
        super(PolicyNetwork, self).__init__()
        # Define network layers (similar structure to DQN reference)
        self.layer1 = nn.Linear(n_observations, 128) # Input layer
        self.layer2 = nn.Linear(128, 128)           # Hidden layer
        self.layer3 = nn.Linear(128, n_actions)      # Output layer (action logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network to get action probabilities.

        Parameters:
        - x (torch.Tensor): Input tensor representing the state(s).

        Returns:
        - torch.Tensor: Output tensor representing action probabilities (after Softmax).
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)

        # Apply layers with ReLU activation for hidden layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Get action logits from the output layer
        action_logits = self.layer3(x)
        # Apply Softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=-1) # Use dim=-1 for generality (works for batches)
        return action_probs

# Action Selection for REINFORCE
def select_action_reinforce(state: torch.Tensor, policy_net: PolicyNetwork) -> Tuple[int, torch.Tensor]:
    """
    Selects an action by sampling from the policy network's output distribution.

    Parameters:
    - state (torch.Tensor): The current state as a tensor of shape [state_dim].
    - policy_net (PolicyNetwork): The policy network used to estimate action probabilities.

    Returns:
    - Tuple[int, torch.Tensor]:
        - action (int): The selected action index.
        - log_prob (torch.Tensor): The log probability of the selected action.
    """
    # Ensure the network is in evaluation mode if it has dropout/batchnorm layers (optional here)
    # policy_net.eval() 

    # Get action probabilities from the policy network
    # Add batch dimension if state is single instance [state_dim] -> [1, state_dim]
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    action_probs = policy_net(state)

    # Create a categorical distribution over the actions
    # Squeeze(0) if we added a batch dimension earlier to get probs for the single state
    m = Categorical(action_probs.squeeze(0)) 
    
    # Sample an action from the distribution
    action = m.sample()
    
    # Get the log probability of the sampled action (needed for gradient calculation)
    log_prob = m.log_prob(action)

    # Put network back to training mode if needed
    # policy_net.train()

    # Return the action index (as int) and its log probability (as tensor)
    return action.item(), log_prob


# Calculating Returns
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> torch.Tensor:
    """
    Calculates the discounted returns G_t for each step t in an episode.

    Parameters:
    - rewards (List[float]): List of rewards received during the episode.
    - gamma (float): The discount factor.
    - standardize (bool): Whether to standardize (normalize) the returns (subtract mean, divide by std).

    Returns:
    - torch.Tensor: A tensor containing the discounted return for each step.
    """
    n_steps = len(rewards)
    returns = torch.zeros(n_steps, device=device, dtype=torch.float32)
    discounted_return = 0.0

    # Iterate backwards through the rewards to calculate discounted returns
    for t in reversed(range(n_steps)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return

    # Standardize returns (optional but often helpful)
    if standardize:
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + 1e-8 # Add small epsilon to prevent division by zero
        returns = (returns - mean_return) / std_return

    return returns


# Optimization Step
def optimize_policy(
    log_probs: List[torch.Tensor], 
    returns: torch.Tensor, 
    optimizer: optim.Optimizer
) -> float:
    """
    Performs one step of optimization on the policy network using REINFORCE update rule.

    Parameters:
    - log_probs (List[torch.Tensor]): List of log probabilities of actions taken in the episode.
    - returns (torch.Tensor): Tensor of discounted returns for each step in the episode.
    - optimizer (optim.Optimizer): The optimizer for updating the policy network.

    Returns:
    - float: The computed loss value for the episode.
    """
    # Stack log probabilities into a single tensor
    log_probs_tensor = torch.stack(log_probs)

    # Calculate the REINFORCE loss: - (returns * log_probs)
    # We want to maximize E[G_t * log(pi)], so we minimize -E[G_t * log(pi)]
    # Sum over the episode steps
    loss = -torch.sum(returns * log_probs_tensor)

    # Perform backpropagation and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update the policy network parameters

    return loss.item()    # Return the loss value for logging


# Hyperparameters for REINFORCE on Custom Grid World
GAMMA_REINFORCE = 0.99         # Discount factor
LR_REINFORCE = 1e-3            # Learning rate (often lower than DQN, sensitive)
NUM_EPISODES_REINFORCE = 1500  # REINFORCE often needs more episodes due to variance
MAX_STEPS_PER_EPISODE_REINFORCE = 200 # Max steps per episode
STANDARDIZE_RETURNS = True     # Whether to standardize returns


# Re-instantiate the custom GridEnvironment
custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10)

# Get the size of the action space and state dimension
n_actions_custom: int = custom_env.get_action_space_size()  # 4 actions
n_observations_custom: int = custom_env.get_state_dimension()  # 2 state dimensions

# Initialize the policy network
policy_net_reinforce: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom).to(device)

# Initialize the optimizer for the policy network
optimizer_reinforce: optim.Adam = optim.Adam(policy_net_reinforce.parameters(), lr=LR_REINFORCE)

# Lists for storing episode statistics for plotting
episode_rewards_reinforce = []
episode_lengths_reinforce = []
episode_losses_reinforce = []



# Training
print("Starting REINFORCE Training on Custom Grid World...")

# Training Loop
for i_episode in range(NUM_EPISODES_REINFORCE):
    # Reset the environment and get the initial state tensor
    state = custom_env.reset()
    
    # Lists to store data for the current episode
    episode_log_probs: List[torch.Tensor] = []
    episode_rewards: List[float] = []
    
    # --- Generate one episode --- 
    for t in range(MAX_STEPS_PER_EPISODE_REINFORCE):
        # Select action based on current policy and store log probability
        action, log_prob = select_action_reinforce(state, policy_net_reinforce)
        episode_log_probs.append(log_prob)
        
        # Take action in the environment
        next_state, reward, done = custom_env.step(action)
        episode_rewards.append(reward)
        
        # Move to the next state
        state = next_state
        
        # Break if the episode finished
        if done:
            break
            
    # --- Episode finished, now update the policy --- 
    
    # Calculate discounted returns for the episode
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    
    # Perform policy optimization
    loss = optimize_policy(episode_log_probs, returns, optimizer_reinforce)
    
    # Store episode statistics
    total_reward = sum(episode_rewards)
    episode_rewards_reinforce.append(total_reward)
    episode_lengths_reinforce.append(t + 1)
    episode_losses_reinforce.append(loss)

    # Print progress periodically (e.g., every 100 episodes)
    if (i_episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards_reinforce[-100:])
        avg_length = np.mean(episode_lengths_reinforce[-100:])
        avg_loss = np.mean(episode_losses_reinforce[-100:])
        print(
            f"Episode {i_episode+1}/{NUM_EPISODES_REINFORCE} | "
            f"Avg Reward (last 100): {avg_reward:.2f} | "
            f"Avg Length: {avg_length:.2f} | "
            f"Avg Loss: {avg_loss:.4f}"
        )

print("Custom Grid World Training Finished (REINFORCE).")


# Plotting results for REINFORCE on Custom Grid World
plt.figure(figsize=(20, 4))

# Rewards
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_reinforce)
plt.title('REINFORCE Custom Grid: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
# Add moving average
rewards_ma_reinforce = np.convolve(episode_rewards_reinforce, np.ones(100)/100, mode='valid')
if len(rewards_ma_reinforce) > 0: 
    plt.plot(np.arange(len(rewards_ma_reinforce)) + 99, rewards_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

# Lengths
plt.subplot(1, 3, 2)
plt.plot(episode_lengths_reinforce)
plt.title('REINFORCE Custom Grid: Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
# Add moving average
lengths_ma_reinforce = np.convolve(episode_lengths_reinforce, np.ones(100)/100, mode='valid')
if len(lengths_ma_reinforce) > 0:
    plt.plot(np.arange(len(lengths_ma_reinforce)) + 99, lengths_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

# Loss
plt.subplot(1, 3, 3)
plt.plot(episode_losses_reinforce)
plt.title('REINFORCE Custom Grid: Episode Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
# Add moving average
losses_ma_reinforce = np.convolve(episode_losses_reinforce, np.ones(100)/100, mode='valid')
if len(losses_ma_reinforce) > 0:
    plt.plot(np.arange(len(losses_ma_reinforce)) + 99, losses_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

plt.tight_layout()
plt.show()


# Analyzing the Learned Policy
def plot_reinforce_policy_grid(policy_net: PolicyNetwork, env: GridEnvironment, device: torch.device) -> None:
    """
    Plots the greedy policy derived from the REINFORCE policy network.
    Note: Shows the most likely action, not a sample.

    Parameters:
    - policy_net (PolicyNetwork): The trained policy network.
    - env (GridEnvironment): The custom grid environment.
    - device (torch.device): The device (CPU/GPU).

    Returns:
    - None: Displays the policy grid plot.
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
                    state_tensor = state_tensor.unsqueeze(0)
                    # Get action probabilities
                    action_probs: torch.Tensor = policy_net(state_tensor)
                    # Select the action with the highest probability (greedy action)
                    best_action: int = action_probs.argmax(dim=1).item()

                policy_grid[r, c] = action_symbols[best_action]
                ax.text(c, r, policy_grid[r, c], ha='center', va='center', color='black', fontsize=12)

    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("REINFORCE Learned Policy (Most Likely Action)")
    plt.show()

# Plot the policy learned by the trained network
print("\nPlotting Learned Policy from REINFORCE:")
plot_reinforce_policy_grid(policy_net_reinforce, custom_env, device)
