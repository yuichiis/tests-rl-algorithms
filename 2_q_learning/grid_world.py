# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizations

# Import type hints
from typing import List, Tuple, Dict, Optional

# set seed for reproducibility
np.random.seed(42)

# Enable inline plotting for Jupyter Notebook
#%matplotlib inline

# Define the GridWorld environment
def create_gridworld(
    rows: int, 
    cols: int, 
    terminal_states: List[Tuple[int, int]], 
    rewards: Dict[Tuple[int, int], int]
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """
    Create a simple GridWorld environment.
    
    Parameters:
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - terminal_states (List[Tuple[int, int]]): List of terminal states as (row, col) tuples.
    - rewards (Dict[Tuple[int, int], int]): Dictionary mapping (row, col) to reward values.
    
    Returns:
    - grid (np.ndarray): A 2D array representing the grid with rewards.
    - state_space (List[Tuple[int, int]]): List of all possible states in the grid.
    - action_space (List[str]): List of possible actions ('up', 'down', 'left', 'right').
    """
    # Initialize the grid with zeros
    grid = np.zeros((rows, cols))
    
    # Assign rewards to specified states
    for (row, col), reward in rewards.items():
        grid[row, col] = reward
    
    # Define the state space as all possible (row, col) pairs
    state_space = [
        (row, col) 
        for row in range(rows) 
        for col in range(cols)
    ]
    
    # Define the action space as the four possible movements
    action_space = ['up', 'down', 'left', 'right']
    
    return grid, state_space, action_space


# Define state transition function
def state_transition(state: Tuple[int, int], action: str, rows: int, cols: int) -> Tuple[int, int]:
    """
    Compute the next state given the current state and action.
    
    Parameters:
    - state (Tuple[int, int]): Current state as (row, col).
    - action (str): Action to take ('up', 'down', 'left', 'right').
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    
    Returns:
    - Tuple[int, int]: The resulting state (row, col) after taking the action.
    """
    # Unpack the current state into row and column
    row, col = state

    # Update the row or column based on the action, ensuring boundaries are respected
    if action == 'up' and row > 0:  # Move up if not in the topmost row
        row -= 1
    elif action == 'down' and row < rows - 1:  # Move down if not in the bottommost row
        row += 1
    elif action == 'left' and col > 0:  # Move left if not in the leftmost column
        col -= 1
    elif action == 'right' and col < cols - 1:  # Move right if not in the rightmost column
        col += 1

    # Return the new state as a tuple
    return (row, col)


# Define reward function
def get_reward(state: Tuple[int, int], rewards: Dict[Tuple[int, int], int]) -> int:
    """
    Get the reward for a given state.

    Parameters:
    - state (Tuple[int, int]): Current state as (row, col).
    - rewards (Dict[Tuple[int, int], int]): Dictionary mapping (row, col) to reward values.

    Returns:
    - int: The reward for the given state. Returns 0 if the state is not in the rewards dictionary.
    """
    # Use the rewards dictionary to fetch the reward for the given state.
    # If the state is not found, return a default reward of 0.
    return rewards.get(state, 0)

# Example usage of the GridWorld environment

# Define the grid dimensions (4x4), terminal states, and rewards
rows, cols = 4, 4  # Number of rows and columns in the grid
terminal_states = [(0, 0), (3, 3)]  # Terminal states with rewards
rewards = {(0, 0): 1, (3, 3): 10}  # Rewards for terminal states

# Create the GridWorld environment
grid, state_space, action_space = create_gridworld(rows, cols, terminal_states, rewards)

# Test the state transition and reward functions
current_state = (2, 2)  # Starting state
action = 'up'  # Action to take
next_state = state_transition(current_state, action, rows, cols)  # Compute the next state
reward = get_reward(next_state, rewards)  # Get the reward for the next state

# Print the results
print("GridWorld:")  # Display the grid with rewards
print(grid)
print(f"Current State: {current_state}")  # Display the current state
print(f"Action Taken: {action}")  # Display the action taken
print(f"Next State: {next_state}")  # Display the resulting next state
print(f"Reward: {reward}")  # Display the reward for the next state


# Initialize Q-table
def initialize_q_table(state_space: List[Tuple[int, int]], action_space: List[str]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Initialize the Q-table with zeros for all state-action pairs.

    Parameters:
    - state_space (List[Tuple[int, int]]): List of all possible states in the environment, represented as (row, col) tuples.
    - action_space (List[str]): List of all possible actions (e.g., 'up', 'down', 'left', 'right').

    Returns:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): A dictionary where each state maps to another dictionary.
      The inner dictionary maps each action to its corresponding Q-value, initialized to 0.
    """
    q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
    for state in state_space:
        # Initialize Q-values for all actions in the given state to 0
        q_table[state] = {action: 0.0 for action in action_space}
    return q_table


# Choose action using epsilon-greedy policy
def choose_action(state: Tuple[int, int], q_table: Dict[Tuple[int, int], Dict[str, float]], action_space: List[str], epsilon: float) -> str:
    """
    Choose an action using the epsilon-greedy policy.

    Parameters:
    - state (Tuple[int, int]): Current state as (row, col).
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - action_space (List[str]): List of possible actions (e.g., 'up', 'down', 'left', 'right').
    - epsilon (float): Exploration rate (0 <= epsilon <= 1).

    Returns:
    - str: The chosen action.
    """
    # With probability epsilon, choose a random action (exploration)
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    # Otherwise, choose the action with the highest Q-value for the current state (exploitation)
    else:
        return max(q_table[state], key=q_table[state].get)

# Update Q-value
def update_q_value(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    state: Tuple[int, int], 
    action: str, 
    reward: int, 
    next_state: Tuple[int, int], 
    alpha: float, 
    gamma: float, 
    action_space: List[str]
) -> None:
    """
    Update the Q-value using the Q-learning update rule.

    Parameters:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - state (Tuple[int, int]): Current state as (row, col).
    - action (str): Action taken.
    - reward (int): Reward received.
    - next_state (Tuple[int, int]): Next state as (row, col).
    - alpha (float): Learning rate (0 < alpha <= 1).
    - gamma (float): Discount factor (0 <= gamma <= 1).
    - action_space (List[str]): List of possible actions.

    Returns:
    - None: Updates the Q-table in place.
    """
    # Get the maximum Q-value for the next state across all possible actions
    max_next_q: float = max(q_table[next_state].values()) if next_state in q_table else 0.0

    # Update the Q-value for the current state-action pair using the Q-learning formula
    q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])

# Run a single episode
def run_episode(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    state_space: List[Tuple[int, int]], 
    action_space: List[str], 
    rewards: Dict[Tuple[int, int], int], 
    rows: int, 
    cols: int, 
    alpha: float, 
    gamma: float, 
    epsilon: float, 
    max_steps: int
) -> int:
    """
    Run a single episode of Q-learning.

    Parameters:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - state_space (List[Tuple[int, int]]): List of all possible states in the environment.
    - action_space (List[str]): List of possible actions (e.g., 'up', 'down', 'left', 'right').
    - rewards (Dict[Tuple[int, int], int]): Dictionary mapping states (row, col) to reward values.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - alpha (float): Learning rate (0 < alpha <= 1).
    - gamma (float): Discount factor (0 <= gamma <= 1).
    - epsilon (float): Exploration rate (0 <= epsilon <= 1).
    - max_steps (int): Maximum number of steps allowed in the episode.

    Returns:
    - int: Total reward accumulated during the episode.
    """
    # Start from a random state
    state: Tuple[int, int] = state_space[np.random.choice(len(state_space))]
    total_reward: int = 0  # Initialize total reward for the episode

    # Loop for a maximum number of steps
    for _ in range(max_steps):
        # Choose an action using the epsilon-greedy policy
        action: str = choose_action(state, q_table, action_space, epsilon)
        
        # Compute the next state based on the chosen action
        next_state: Tuple[int, int] = state_transition(state, action, rows, cols)
        
        # Get the reward for the next state
        reward: int = get_reward(next_state, rewards)
        
        # Update the Q-value for the current state-action pair
        update_q_value(q_table, state, action, reward, next_state, alpha, gamma, action_space)
        
        # Accumulate the reward
        total_reward += reward
        
        # Move to the next state
        state = next_state
        
        # Check if the agent has reached a terminal state
        if state in terminal_states:
            break
    
    # Return the total reward accumulated during the episode
    return total_reward

# Set hyperparameters for the Q-learning algorithm
alpha = 0.1  # Learning rate: Determines how much new information overrides old information
gamma = 0.9  # Discount factor: Determines the importance of future rewards
epsilon = 0.1  # Exploration rate: Probability of choosing a random action (exploration vs. exploitation)
max_steps = 100  # Maximum number of steps allowed per episode
episodes = 500  # Total number of episodes to run

# Initialize the Q-table with zeros for all state-action pairs
q_table = initialize_q_table(state_space, action_space)

# List to store the total reward accumulated in each episode
rewards_per_episode = []

# Run multiple episodes of Q-learning
for episode in range(episodes):
    # Run a single episode and get the total reward
    total_reward = run_episode(q_table, state_space, action_space, rewards, rows, cols, alpha, gamma, epsilon, max_steps)
    # Append the total reward for this episode to the rewards list
    rewards_per_episode.append(total_reward)

# Adjust figure size for better visibility
plt.figure(figsize=(20, 3))

# Plot the total rewards accumulated over episodes
plt.plot(rewards_per_episode)
plt.xlabel('Episode')  # Label for the x-axis
plt.ylabel('Total Reward')  # Label for the y-axis
plt.title('Rewards Over Episodes')  # Title of the plot
plt.show()  # Display the plot


# Initialize Q-table
def initialize_q_table(state_space, action_space):
    """
    Initialize the Q-table with zeros.
    
    Parameters:
    - state_space: List of all possible states.
    - action_space: List of possible actions.
    
    Returns:
    - q_table: A dictionary mapping state-action pairs to Q-values.
    """
    q_table = {}
    for state in state_space:
        q_table[state] = {action: 0 for action in action_space}
    return q_table

# Choose action using epsilon-greedy policy
def choose_action(state, q_table, action_space, epsilon):
    """
    Choose an action using the epsilon-greedy policy.
    
    Parameters:
    - state: Current state as (row, col).
    - q_table: Q-table mapping state-action pairs to Q-values.
    - action_space: List of possible actions.
    - epsilon: Exploration rate.
    
    Returns:
    - action: The chosen action.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)  # Explore
    else:
        return max(q_table[state], key=q_table[state].get)  # Exploit

# Update Q-value
def update_q_value(q_table, state, action, reward, next_state, alpha, gamma, action_space):
    """
    Update the Q-value using the Q-learning update rule.
    
    Parameters:
    - q_table: Q-table mapping state-action pairs to Q-values.
    - state: Current state as (row, col).
    - action: Action taken.
    - reward: Reward received.
    - next_state: Next state as (row, col).
    - alpha: Learning rate.
    - gamma: Discount factor.
    - action_space: List of possible actions.
    
    Returns:
    - None (updates q_table in place).
    """
    max_next_q = max(q_table[next_state].values()) if next_state in q_table else 0
    q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])

# Define epsilon-greedy policy
def epsilon_greedy_policy(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    state: Tuple[int, int], 
    action_space: List[str], 
    epsilon: float
) -> str:
    """
    Implement the epsilon-greedy policy for action selection.
    
    Parameters:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - state (Tuple[int, int]): Current state as (row, col).
    - action_space (List[str]): List of possible actions.
    - epsilon (float): Exploration rate.
    
    Returns:
    - str: The chosen action.
    """
    # With probability epsilon, choose a random action (exploration)
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    # Otherwise, choose the action with the highest Q-value for the current state (exploitation)
    else:
        return max(q_table[state], key=q_table[state].get)
    
# Define dynamic epsilon adjustment
def adjust_epsilon(
    initial_epsilon: float, 
    min_epsilon: float, 
    decay_rate: float, 
    episode: int
) -> float:
    """
    Dynamically adjust epsilon over time to balance exploration and exploitation.
    
    Parameters:
    - initial_epsilon (float): Initial exploration rate.
    - min_epsilon (float): Minimum exploration rate.
    - decay_rate (float): Rate at which epsilon decays.
    - episode (int): Current episode number.
    
    Returns:
    - float: Adjusted exploration rate.
    """
    # Compute the decayed epsilon value, ensuring it doesn't go below the minimum epsilon
    return max(min_epsilon, initial_epsilon * np.exp(-decay_rate * episode))

# Example usage of epsilon-greedy policy and dynamic epsilon adjustment
initial_epsilon: float = 1.0  # Start with full exploration
min_epsilon: float = 0.1  # Minimum exploration rate
decay_rate: float = 0.01  # Decay rate for epsilon
episodes: int = 500  # Number of episodes

# Track epsilon values over episodes
epsilon_values: List[float] = []
for episode in range(episodes):
    # Adjust epsilon for the current episode
    epsilon = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)
    epsilon_values.append(epsilon)

# Adjust figure size for better visibility
plt.figure(figsize=(20, 3))

# Plot epsilon decay over episodes
plt.plot(epsilon_values)
plt.xlabel('Episode')  # Label for the x-axis
plt.ylabel('Epsilon')  # Label for the y-axis
plt.title('Epsilon Decay Over Episodes')  # Title of the plot
plt.show()  # Display the plot

# Execute the Q-Learning algorithm over multiple episodes and track performance metrics
def run_q_learning(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    state_space: List[Tuple[int, int]], 
    action_space: List[str], 
    rewards: Dict[Tuple[int, int], int], 
    rows: int, 
    cols: int, 
    alpha: float, 
    gamma: float, 
    initial_epsilon: float, 
    min_epsilon: float, 
    decay_rate: float, 
    episodes: int, 
    max_steps: int
) -> Tuple[List[int], List[int]]:
    """
    Execute the Q-Learning algorithm over multiple episodes.

    Parameters:
    - q_table: Q-table mapping state-action pairs to Q-values.
    - state_space: List of all possible states.
    - action_space: List of possible actions.
    - rewards: Dictionary mapping (row, col) to reward values.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - alpha: Learning rate.
    - gamma: Discount factor.
    - initial_epsilon: Initial exploration rate.
    - min_epsilon: Minimum exploration rate.
    - decay_rate: Rate at which epsilon decays.
    - episodes: Number of episodes to run.
    - max_steps: Maximum number of steps per episode.
    
    Returns:
    - rewards_per_episode: List of total rewards per episode.
    - episode_lengths: List of episode lengths.
    """
    # Initialize lists to store metrics
    rewards_per_episode: List[int] = []
    episode_lengths: List[int] = []
    
    # Loop through each episode
    for episode in range(episodes):
        # Start from a random state
        state: Tuple[int, int] = state_space[np.random.choice(len(state_space))]
        total_reward: int = 0  # Initialize total reward for the episode
        steps: int = 0  # Initialize step counter
        # Adjust epsilon for the current episode
        epsilon: float = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)
        
        # Loop for a maximum number of steps
        for _ in range(max_steps):
            # Choose an action using the epsilon-greedy policy
            action: str = epsilon_greedy_policy(q_table, state, action_space, epsilon)
            # Compute the next state based on the chosen action
            next_state: Tuple[int, int] = state_transition(state, action, rows, cols)
            # Get the reward for the next state
            reward: int = get_reward(next_state, rewards)
            # Update the Q-value for the current state-action pair
            update_q_value(q_table, state, action, reward, next_state, alpha, gamma, action_space)
            # Accumulate the reward
            total_reward += reward
            # Move to the next state
            state = next_state
            # Increment the step counter
            steps += 1
            # Check if the agent has reached a terminal state
            if state in terminal_states:
                break
        
        # Append metrics for the current episode
        rewards_per_episode.append(total_reward)
        episode_lengths.append(steps)
    
    # Return the metrics
    return rewards_per_episode, episode_lengths

# Set hyperparameters for Q-Learning
alpha: float = 0.1  # Learning rate
gamma: float = 0.9  # Discount factor
initial_epsilon: float = 1.0  # Initial exploration rate
min_epsilon: float = 0.1  # Minimum exploration rate
decay_rate: float = 0.01  # Decay rate for epsilon
episodes: int = 500  # Number of episodes
max_steps: int = 100  # Maximum steps per episode

# Initialize the Q-table
q_table: Dict[Tuple[int, int], Dict[str, float]] = initialize_q_table(state_space, action_space)

# Execute the Q-Learning algorithm
rewards_per_episode, episode_lengths = run_q_learning(
    q_table, state_space, action_space, rewards, rows, cols, alpha, gamma,
    initial_epsilon, min_epsilon, decay_rate, episodes, max_steps
)

# Plot cumulative rewards over episodes
plt.figure(figsize=(20, 3))

# Plot total rewards per episode
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')  # Label for the x-axis
plt.ylabel('Total Reward')  # Label for the y-axis
plt.title('Cumulative Rewards Over Episodes')  # Title of the plot

# Plot episode lengths per episode
plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.xlabel('Episode')  # Label for the x-axis
plt.ylabel('Episode Length')  # Label for the y-axis
plt.title('Episode Lengths Over Episodes')  # Title of the plot

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Function to visualize the Q-value heatmap
def plot_q_values(q_table: Dict[Tuple[int, int], Dict[str, float]], rows: int, cols: int, action_space: List[str]) -> None:
    """
    Visualize the Q-values as a heatmap for each action.

    Parameters:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - action_space (List[str]): List of possible actions.

    Returns:
    - None: Displays heatmaps for Q-values of each action.
    """
    # Create subplots for each action
    fig, axes = plt.subplots(1, len(action_space), figsize=(15, 5))
    for i, action in enumerate(action_space):
        # Initialize a grid to store Q-values for the current action
        q_values = np.zeros((rows, cols))
        for (row, col), actions in q_table.items():
            q_values[row, col] = actions[action]  # Extract Q-value for the current action

        # Plot the heatmap for the current action
        ax = axes[i]
        cax = ax.matshow(q_values, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Q-values for action: {action}")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")

    # Adjust layout and display the heatmaps
    plt.tight_layout()
    plt.show()

# Function to visualize the learned policy
def plot_policy(q_table: Dict[Tuple[int, int], Dict[str, float]], rows: int, cols: int) -> None:
    """
    Visualize the learned policy as arrows on the grid.

    Parameters:
    - q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table mapping state-action pairs to Q-values.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.

    Returns:
    - None: Displays the policy visualization.
    """
    # Initialize a grid to store the best action for each state
    policy_grid = np.empty((rows, cols), dtype=str)
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}  # Symbols for actions

    # Determine the best action for each state based on Q-values
    for (row, col), actions in q_table.items():
        best_action = max(actions, key=actions.get)  # Get the action with the highest Q-value
        policy_grid[row, col] = action_symbols[best_action]  # Map the action to its symbol

    # Plot the policy grid with increased width
    fig, ax = plt.subplots(figsize=(16, 3))  # Increased width from 12 to 16 for more horizontal stretch
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=14)  # Slightly larger font
    
    # Create a wider grid with more horizontal space
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)  # Add a faint background grid
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Learned Policy")
    plt.tight_layout()
    plt.show()

# Plot the Q-value heatmaps and the learned policy side by side
fig, axes = plt.subplots(1, len(action_space) + 1, figsize=(20, 5))

# Plot the Q-value heatmaps for each action
for i, action in enumerate(action_space):
    q_values = np.zeros((rows, cols))
    for (row, col), actions in q_table.items():
        q_values[row, col] = actions[action]
    cax = axes[i].matshow(q_values, cmap='viridis')
    fig.colorbar(cax, ax=axes[i])
    axes[i].set_title(f"Q-values for action: {action}")
    axes[i].set_xlabel("Columns")
    axes[i].set_ylabel("Rows")

# Plot the learned policy
policy_grid = np.empty((rows, cols), dtype=str)
action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
for (row, col), actions in q_table.items():
    best_action = max(actions, key=actions.get)
    policy_grid[row, col] = action_symbols[best_action]

axes[-1].matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
for i in range(rows):
    for j in range(cols):
        axes[-1].text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=14)
axes[-1].set_title("Learned Policy")
axes[-1].set_xlabel("Columns")
axes[-1].set_ylabel("Rows")

plt.tight_layout()
plt.show()

# Create a list of dictionaries to represent the Q-table data
q_policy_data = []
for state, actions in q_table.items():
    # Append a dictionary for each state containing Q-values for all actions and the optimal action
    q_policy_data.append({
        'State': state,  # The current state (row, col)
        'up': actions['up'],  # Q-value for the 'up' action
        'down': actions['down'],  # Q-value for the 'down' action
        'left': actions['left'],  # Q-value for the 'left' action
        'right': actions['right'],  # Q-value for the 'right' action
        'Optimal Action': max(actions, key=actions.get)  # The action with the highest Q-value
    })

# Display the Q-table data in a tabular format
header = ['State', 'up', 'down', 'left', 'right', 'Optimal Action']  # Define the table headers
# Print the table header with proper spacing
print(f"{header[0]:<10} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<10} {header[5]:<15}")
print("-" * 65)  # Print a separator line for better readability

# Iterate through the Q-table data and print each row
for row in q_policy_data:
    # Print the state, Q-values for all actions, and the optimal action
    print(f"{row['State']!s:<10} {row['up']:<10.2f} {row['down']:<10.2f} {row['left']:<10.2f} {row['right']:<10.2f} {row['Optimal Action']:<15}")

# Experiment with different hyperparameters
learning_rates = [0.1, 0.5]  # Different learning rates (alpha) to test
discount_factors = [0.5]  # Different discount factors (gamma) to test
exploration_rates = [1.0]  # Different initial exploration rates (epsilon) to test

# Store results for comparison
results = []

# Run experiments with different hyperparameter combinations
for alpha in learning_rates:  # Iterate over different learning rates
    for gamma in discount_factors:  # Iterate over different discount factors
        for initial_epsilon in exploration_rates:  # Iterate over different initial exploration rates
            # Initialize Q-table for the current experiment
            q_table = initialize_q_table(state_space, action_space)
            
            # Run Q-Learning with the current set of hyperparameters
            rewards_per_episode, episode_lengths = run_q_learning(
                q_table, state_space, action_space, rewards, rows, cols, alpha, gamma,
                initial_epsilon, min_epsilon, decay_rate, episodes, max_steps
            )

            # Store the results of the current experiment
            results.append({
                'alpha': alpha,  # Learning rate
                'gamma': gamma,  # Discount factor
                'initial_epsilon': initial_epsilon,  # Initial exploration rate
                'rewards_per_episode': rewards_per_episode,  # Rewards collected per episode
                'episode_lengths': episode_lengths  # Length of each episode
            })

# Create a larger figure to visualize all hyperparameter combinations
plt.figure(figsize=(20, 5))

# Calculate the number of rows and columns for the subplot grid
num_rows = len(learning_rates)  # Number of rows corresponds to the number of learning rates
num_cols = len(discount_factors) * len(exploration_rates)  # Number of columns corresponds to combinations of discount factors and exploration rates

# Plot the results of each experiment
for i, result in enumerate(results):  # Iterate over all results
    plt.subplot(num_rows, num_cols, i + 1)  # Create a subplot for each experiment
    plt.plot(result['rewards_per_episode'])  # Plot rewards per episode
    plt.title(f"α={result['alpha']}, γ={result['gamma']}, ε={result['initial_epsilon']}")  # Add a title with hyperparameter values
    plt.xlabel('Episode')  # Label for the x-axis
    plt.ylabel('Total Reward')  # Label for the y-axis

# Adjust layout to prevent overlap and display the plots
plt.tight_layout()
plt.show()

# Define the Cliff Walking environment 
def create_cliff_walking_env(
    rows: int, 
    cols: int, 
    cliff_states: List[Tuple[int, int]], 
    terminal_state: Tuple[int, int], 
    rewards: Dict[Tuple[int, int], int]
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """
    Create a Cliff Walking environment.

    Parameters:
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - cliff_states (List[Tuple[int, int]]): List of cliff states as (row, col) tuples.
    - terminal_state (Tuple[int, int]): The terminal state as a (row, col) tuple.
    - rewards (Dict[Tuple[int, int], int]): Dictionary mapping (row, col) to reward values.

    Returns:
    - Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
        - grid (np.ndarray): A 2D array representing the grid with rewards.
        - state_space (List[Tuple[int, int]]): List of all possible states in the grid.
        - action_space (List[str]): List of possible actions ('up', 'down', 'left', 'right').
    """
    # Initialize the grid with zeros
    grid = np.zeros((rows, cols))

    # Assign rewards to specified states
    for (row, col), reward in rewards.items():
        grid[row, col] = reward

    # Assign a high negative reward for cliff states
    for row, col in cliff_states:
        grid[row, col] = -100

    # Assign a positive reward for the terminal state
    grid[terminal_state] = 10

    # Define the state space as all possible (row, col) pairs
    state_space = [(r, c) for r in range(rows) for c in range(cols)]

    # Define the action space as the four possible movements
    action_space = ['up', 'down', 'left', 'right']

    return grid, state_space, action_space

# Define the Cliff Walking environment
rows, cols = 4, 12  # Dimensions of the grid (4 rows and 12 columns)

# Define the cliff states (bottom row, excluding start and goal positions)
cliff_states = [(3, c) for c in range(1, 11)]  

# Define the terminal state (goal position)
terminal_state = (3, 11)

# Define the rewards for the environment
# Reward for reaching the terminal state is 10
rewards = {(3, 11): 10}  

# Create the Cliff Walking environment using the helper function
# This function returns the grid, state space, and action space
cliff_grid, cliff_state_space, cliff_action_space = create_cliff_walking_env(
    rows, cols, cliff_states, terminal_state, rewards
)

# Plot rewards for the Cliff Walking environment
def plot_rewards(rewards_per_episode: List[int], ax: plt.Axes = None) -> plt.Axes:
    """
    Plot the total rewards accumulated over episodes.

    Parameters:
    - rewards_per_episode (List[int]): List of total rewards per episode.
    - ax (plt.Axes, optional): Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns:
    - plt.Axes: The Matplotlib axis containing the plot.
    """
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the rewards over episodes
    ax.plot(rewards_per_episode)
    ax.set_xlabel('Episode')  # Label for the x-axis
    ax.set_ylabel('Total Reward')  # Label for the y-axis
    ax.set_title('Rewards Over Episodes')  # Title of the plot
    
    # Return the axis for further customization if needed
    return ax

# Visualize the Cliff Walking environment
def plot_cliff_walking_env(
    grid: np.ndarray, 
    cliff_states: List[Tuple[int, int]], 
    terminal_state: Tuple[int, int], 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize the Cliff Walking environment.

    Parameters:
    - grid: 2D numpy array representing the grid.
    - cliff_states: List of cliff states as (row, col) tuples.
    - terminal_state: The terminal state as a (row, col) tuple.
    - ax: Optional Matplotlib axis to plot on.

    Returns:
    - ax: Matplotlib axis with the environment visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) in cliff_states:
                ax.text(c, r, 'C', ha='center', va='center', color='red', fontsize=12)
            elif (r, c) == terminal_state:
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=12)
            elif (r, c) == (3, 0):  # Starting position
                ax.text(c, r, 'S', ha='center', va='center', color='blue', fontsize=12)
            else:
                ax.text(c, r, '.', ha='center', va='center', color='black', fontsize=12)
    
    ax.matshow(np.zeros_like(grid), cmap='Greys', alpha=0.1)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Cliff Walking Environment")
    return ax

# Visualize the Q-values for each action
def plot_q_values(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    rows: int, 
    cols: int, 
    action_space: List[str], 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize the Q-values for each action.

    Parameters:
    - q_table: Dictionary containing Q-values for each state-action pair.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - action_space: List of possible actions.
    - ax: Optional Matplotlib axis to plot on.

    Returns:
    - ax: Matplotlib axis with the Q-values visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a grid for Q-values
    q_values = np.zeros((rows, cols))
    
    # For each state, find the action with the highest Q-value
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            if state in q_table:
                q_values[r, c] = max(q_table[state].values())
    
    im = ax.imshow(q_values, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Max Q-Value')
    
    # Add state values as text
    for r in range(rows):
        for c in range(cols):
            if (r, c) in cliff_states:
                text_color = 'red'
            elif (r, c) == terminal_state:
                text_color = 'green'
            else:
                text_color = 'white'
            ax.text(c, r, f"{q_values[r, c]:.1f}", ha='center', va='center', color=text_color)
    
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title('Max Q-Values')
    return ax

# Visualize the policy derived from the Q-table
def plot_policy(
    q_table: Dict[Tuple[int, int], Dict[str, float]], 
    rows: int, 
    cols: int, 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize the policy derived from the Q-table.

    Parameters:
    - q_table: Dictionary containing Q-values for each state-action pair.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - ax: Optional Matplotlib axis to plot on.

    Returns:
    - ax: Matplotlib axis with the policy visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define symbols for actions
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    
    # Create a grid for the policy
    policy_grid = np.empty((rows, cols), dtype='U1')
    
    # For each state, find the action with the highest Q-value
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            if state in q_table:
                best_action = max(q_table[state], key=q_table[state].get)
                policy_grid[r, c] = action_symbols[best_action]
            else:
                policy_grid[r, c] = ' '
    
    # Display the policy grid
    ax.imshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    
    # Add policy arrows as text
    for r in range(rows):
        for c in range(cols):
            if (r, c) in cliff_states:
                text_color = 'red'
            elif (r, c) == terminal_state:
                text_color = 'green'
            else:
                text_color = 'black'
            ax.text(c, r, policy_grid[r, c], ha='center', va='center', color=text_color, fontsize=20)
    
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title('Learned Policy')
    return ax

# Run Q-Learning on the Cliff Walking environment
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
initial_epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.1  # Minimum exploration rate
decay_rate = 0.01  # Decay rate for epsilon
episodes = 500  # Number of episodes
max_steps = 100  # Maximum steps per episode

# Initialize Q-table
cliff_q_table = initialize_q_table(cliff_state_space, cliff_action_space)

# Execute Q-Learning
cliff_rewards_per_episode, cliff_episode_lengths = run_q_learning(
    cliff_q_table, cliff_state_space, cliff_action_space, rewards, rows, cols, alpha, gamma,
    initial_epsilon, min_epsilon, decay_rate, episodes, max_steps
)

# Create a 2x2 grid of visualizations
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Plot rewards in the top-left subplot
plot_rewards(cliff_rewards_per_episode, ax=axs[0, 0])

# Plot environment in the top-right subplot
plot_cliff_walking_env(cliff_grid, cliff_states, terminal_state, ax=axs[0, 1])

# Plot Q-values in the bottom-left subplot
plot_q_values(cliff_q_table, rows, cols, cliff_action_space, ax=axs[1, 0])

# Plot policy in the bottom-right subplot
plot_policy(cliff_q_table, rows, cols, ax=axs[1, 1])

plt.tight_layout()
plt.show()