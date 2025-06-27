# Import necessary libraries for numerical computations, plotting, and utility functions
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional

# Import TensorFlow for building and training neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber

# Set up the device to use GPU if available, otherwise fallback to CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Using {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        device = "/GPU:0"
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        device = "/CPU:0"
else:
    device = "/CPU:0"
print(f"Using device: {device}")


# Set a random seed for reproducibility across runs
seed = 42
random.seed(seed)  # Seed for Python's random module
np.random.seed(seed)  # Seed for NumPy
tf.random.set_seed(seed) # Seed for TensorFlow

# Enable inline plotting for Jupyter Notebook
#%matplotlib inline


# <<< 変更 >>> Custom Grid World Environmentを迷路対応に修正 (戻り値をNumPy配列に変更)
class GridEnvironment:
    """
    A simple 10x10 Grid World environment with walls (a maze).
    State: (row, col) represented as normalized vector [row/10, col/10].
    Actions: 0 (up), 1 (down), 2 (left), 3 (right).
    Rewards: +10 for reaching the goal, -1 for hitting a wall, -0.1 for each step.
    Action selection is restricted by walls.
    """

    def __init__(self, rows: int = 10, cols: int = 10, maze: Optional[np.ndarray] = None) -> None:
        self.rows: int = rows
        self.cols: int = cols

        if maze is None:
            self.maze: np.ndarray = np.zeros((rows, cols), dtype=int)
        else:
            self.maze: np.ndarray = maze
            assert self.maze.shape == (rows, cols), "Maze dimensions must match rows and cols"

        self.start_state: Tuple[int, int] = (0, 0)
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
        self.state: Tuple[int, int] = self.start_state
        self.state_dim: int = 2
        self.action_dim: int = 4

        self.action_map: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the start state. Returns a NumPy array.
        """
        self.state = self.start_state
        return self._get_state_array(self.state)

    def _get_state_array(self, state_tuple: Tuple[int, int]) -> np.ndarray:
        """
        Converts a (row, col) tuple to a normalized NumPy array for the network.
        """
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1),
            state_tuple[1] / (self.cols - 1)
        ]
        return np.array(normalized_state, dtype=np.float32)

    def _get_state_tuple_from_array(self, state_array: np.ndarray) -> Tuple[int, int]:
        """Converts a normalized state array back to a (row, col) tuple."""
        row = int(round(state_array[0] * (self.rows - 1)))
        col = int(round(state_array[1] * (self.cols - 1)))
        return (row, col)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Performs one step in the environment based on the given action.
        """
        if self.state == self.goal_state:
            return self._get_state_array(self.state), 0.0, True

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1  # Default step cost

        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or self.maze[next_row, next_col] == 1:
            next_row, next_col = current_row, current_col
            reward = -1.0
        
        self.state = (next_row, next_col)
        next_state_array: np.ndarray = self._get_state_array(self.state)

        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0

        return next_state_array, reward, done

    def get_available_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        if state is None:
            state = self.state
        
        if state == self.goal_state:
            return []

        available_actions = []
        current_row, current_col = state

        for action, (dr, dc) in self.action_map.items():
            next_row, next_col = current_row + dr, current_col + dc
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols and self.maze[next_row, next_col] == 0:
                available_actions.append(action)
        
        return available_actions

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

# <<< 追加 >>> 迷路の定義 (0: 道, 1: 壁)
default_maze = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
])

# Instantiate the maze environment
custom_env = GridEnvironment(rows=10, cols=10, maze=default_maze)
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

print(f"Custom Maze Environment:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"Start State: {custom_env.start_state}, Goal State: {custom_env.goal_state}")
print(f"Available actions at start state (0,0): {custom_env.get_available_actions((0,0))}")
print(f"Available actions at (2,1): {custom_env.get_available_actions((2,1))}")


# <<< 変更 >>> Define the Q-Network architecture using TensorFlow/Keras
def create_dqn_model(n_observations: int, n_actions: int) -> tf.keras.Model:
    """Creates a Sequential Keras model for the DQN."""
    model = Sequential([
        InputLayer(input_shape=(n_observations,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(n_actions, activation='linear') # Output layer has a linear activation
    ])
    return model

# Replay Memory (変更なし)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)
    def __len__(self) -> int:
        return len(self.memory)


# <<< 変更 >>> Action Selectionを迷路対応に修正 (TensorFlow/NumPy版)
def select_action_custom(state: np.ndarray,
                         policy_net: tf.keras.Model,
                         env: GridEnvironment,
                         epsilon_start: float,
                         epsilon_end: float,
                         epsilon_decay: int) -> Tuple[np.ndarray, float]:
    """
    Selects an action using epsilon-greedy, restricted to available actions.
    """
    global steps_done_custom
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done_custom / epsilon_decay)
    steps_done_custom += 1

    current_state_tuple = env._get_state_tuple_from_array(state)
    available_actions = env.get_available_actions(current_state_tuple)

    if not available_actions:
        return np.array([[0]], dtype=np.int64), epsilon_threshold

    if sample > epsilon_threshold:
        # Exploitation: Select the best action among available ones
        state_batch = np.expand_dims(state, axis=0)
        # Use training=False for inference
        q_values = policy_net(state_batch, training=False)[0].numpy()
        
        mask = np.full(env.get_action_space_size(), -np.inf)
        mask[available_actions] = 0.0
        masked_q_values = q_values + mask
        
        action = np.argmax(masked_q_values).reshape(1, 1)
    else:
        # Exploration: Select a random action from available ones
        action = np.array([[random.choice(available_actions)]], dtype=np.int64)

    return action, epsilon_threshold

# <<< 変更 >>> Optimization Stepを迷路対応に修正 (TensorFlow/GradientTape版)
@tf.function # Compile into a TensorFlow graph for performance
def optimize_model(memory: ReplayMemory,
                   policy_net: tf.keras.Model,
                   target_net: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   env: GridEnvironment,
                   batch_size: int,
                   gamma: float,
                   criterion) -> Optional[float]:
    """
    Performs one step of optimization using tf.GradientTape.
    """
    # This check cannot be inside @tf.function, so we handle it outside
    # if len(memory) < batch_size:
    #     return None
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = np.array([s is not None for s in batch.next_state])
    non_final_next_states = tf.constant(
        [s for s in batch.next_state if s is not None], dtype=tf.float32
    )

    state_batch = tf.constant(batch.state, dtype=tf.float32)
    action_batch = tf.constant(batch.action, dtype=tf.int64)
    reward_batch = tf.constant(batch.reward, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Compute Q(s_t, a)
        state_action_values = tf.gather(
            policy_net(state_batch, training=True), action_batch, batch_dims=1
        )

        # Compute V(s_{t+1}) for all next states.
        next_state_values = tf.zeros(batch_size, dtype=tf.float32)
        if tf.reduce_sum(tf.cast(non_final_mask, tf.float32)) > 0:
            target_q_values = target_net(non_final_next_states, training=False)
            
            # Mask unavailable actions in the next states
            next_state_tuples = [env._get_state_tuple_from_array(s.numpy()) for s in non_final_next_states]
            
            mask = np.full(target_q_values.shape, -np.inf, dtype=np.float32)
            for i, state_tuple in enumerate(next_state_tuples):
                available_actions = env.get_available_actions(state_tuple)
                if available_actions:
                    mask[i, available_actions] = 0.0
            
            masked_q_values = target_q_values + tf.constant(mask, dtype=tf.float32)
            max_q_values = tf.reduce_max(masked_q_values, axis=1)

            # Update the next_state_values tensor
            indices = tf.where(non_final_mask)
            next_state_values = tf.tensor_scatter_nd_update(next_state_values, indices, max_q_values)

        # Compute the expected Q values (TD Target)
        expected_state_action_values = (next_state_values * gamma) + tf.squeeze(reward_batch)

        # Compute loss
        loss = criterion(tf.expand_dims(expected_state_action_values, 1), state_action_values)

    # Compute gradients and update weights
    grads = tape.gradient(loss, policy_net.trainable_variables)
    # Clip gradients
    clipped_grads = [tf.clip_by_value(grad, -100.0, 100.0) for grad in grads]
    optimizer.apply_gradients(zip(clipped_grads, policy_net.trainable_variables))

    return loss

# <<< 変更 >>> Target Network Update (TensorFlow版)
def update_target_net(policy_net: tf.keras.Model, target_net: tf.keras.Model) -> None:
    target_net.set_weights(policy_net.get_weights())


# <<< 変更 >>> ハイパーパラメータを迷路用に調整
BATCH_SIZE_CUSTOM = 128
GAMMA_CUSTOM = 0.99
EPS_START_CUSTOM = 1.0
EPS_END_CUSTOM = 0.05
EPS_DECAY_CUSTOM = 20000
LR_CUSTOM = 5e-4
MEMORY_CAPACITY_CUSTOM = 20000
TARGET_UPDATE_FREQ_CUSTOM = 20
NUM_EPISODES_CUSTOM = 1000
MAX_STEPS_PER_EPISODE_CUSTOM = 500

# Re-instantiate environment and networks
custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10, maze=default_maze)
n_actions_custom: int = custom_env.get_action_space_size()
n_observations_custom: int = custom_env.get_state_dimension()

# Create policy and target networks
policy_net_custom = create_dqn_model(n_observations_custom, n_actions_custom)
target_net_custom = create_dqn_model(n_observations_custom, n_actions_custom)
update_target_net(policy_net_custom, target_net_custom)

# Setup optimizer, loss function, and replay memory
optimizer_custom = AdamW(learning_rate=LR_CUSTOM, amsgrad=True)
criterion_custom = Huber()
memory_custom: ReplayMemory = ReplayMemory(MEMORY_CAPACITY_CUSTOM)

episode_rewards_custom = []
episode_lengths_custom = []
episode_epsilons_custom = []
episode_losses_custom = []

# Training Loop
print("Starting DQN Training on Custom Maze World (TensorFlow)...")
steps_done_custom = 0

for i_episode in range(NUM_EPISODES_CUSTOM):
    state = custom_env.reset()
    total_reward = 0
    current_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_CUSTOM):
        action_array, current_epsilon = select_action_custom(
            state, policy_net_custom, custom_env, EPS_START_CUSTOM, EPS_END_CUSTOM, EPS_DECAY_CUSTOM
        )
        action = action_array.item()

        next_state, reward, done = custom_env.step(action)
        total_reward += reward

        # Store transition as NumPy arrays
        memory_next_state = next_state if not done else None
        memory_custom.push(state, action_array, memory_next_state, np.array([reward], dtype=np.float32), np.array([done]))

        state = next_state

        # Perform one step of optimization
        if len(memory_custom) >= BATCH_SIZE_CUSTOM:
            loss = optimize_model(
                memory_custom, policy_net_custom, target_net_custom, optimizer_custom, custom_env, BATCH_SIZE_CUSTOM, GAMMA_CUSTOM, criterion_custom
            )
            if loss is not None:
                current_losses.append(loss.numpy())

        if done:
            break

    episode_rewards_custom.append(total_reward)
    episode_lengths_custom.append(t + 1)
    episode_epsilons_custom.append(current_epsilon)
    episode_losses_custom.append(np.mean(current_losses) if current_losses else 0)

    if i_episode % TARGET_UPDATE_FREQ_CUSTOM == 0:
        update_target_net(policy_net_custom, target_net_custom)

    if (i_episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards_custom[-50:])
        avg_length = np.mean(episode_lengths_custom[-50:])
        avg_loss = np.mean([l for l in episode_losses_custom[-50:] if l > 0])
        print(
            f"Episode {i_episode+1}/{NUM_EPISODES_CUSTOM} | "
            f"Avg Reward (last 50): {avg_reward:.2f} | "
            f"Avg Length: {avg_length:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {current_epsilon:.3f}"
        )

print("Custom Maze World Training Finished.")


# Plotting results (変更なし)
plt.figure(figsize=(20, 3))
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_custom)
plt.title('DQN Maze: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
rewards_ma_custom = np.convolve(episode_rewards_custom, np.ones(50)/50, mode='valid')
if len(rewards_ma_custom) > 0:
    plt.plot(np.arange(len(rewards_ma_custom)) + 49, rewards_ma_custom, label='50-episode MA', color='orange')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(episode_lengths_custom)
plt.title('DQN Maze: Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
lengths_ma_custom = np.convolve(episode_lengths_custom, np.ones(50)/50, mode='valid')
if len(lengths_ma_custom) > 0:
    plt.plot(np.arange(len(lengths_ma_custom)) + 49, lengths_ma_custom, label='50-episode MA', color='orange')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(episode_epsilons_custom)
plt.title('DQN Maze: Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)

plt.tight_layout()
plt.show()

# <<< 変更 >>> Analyzing the Learned Policy (迷路対応, TensorFlow版)
def plot_dqn_policy_grid(policy_net: tf.keras.Model, env: GridEnvironment) -> None:
    """
    Plots the greedy policy derived from the DQN, including maze walls.
    """
    rows: int = env.rows
    cols: int = env.cols
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.7))

    maze_display = env.maze.astype(float)
    ax.imshow(maze_display, cmap='Greys', interpolation='nearest')

    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)

            if env.maze[r, c] == 1:
                continue
            if state_tuple == env.goal_state:
                ax.text(c, r, 'G', ha='center', va='center', color='lime', fontsize=16, weight='bold')
                continue
            
            available_actions = env.get_available_actions(state_tuple)
            if not available_actions:
                continue

            state_array: np.ndarray = env._get_state_array(state_tuple)
            state_batch = np.expand_dims(state_array, axis=0)
            
            # Get Q-values from the network
            q_values: np.ndarray = policy_net(state_batch, training=False).numpy()[0]
            
            # Select the best action among available ones
            best_action = -1
            max_q = -float('inf')
            for action in available_actions:
                if q_values[action] > max_q:
                    max_q = q_values[action]
                    best_action = action
            
            if best_action != -1:
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='red', fontsize=12)

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DQN Learned Policy (Maze World - TensorFlow)")
    plt.show()

# Plot the policy learned by the trained network
print("\nPlotting Learned Policy from DQN (TensorFlow):")
plot_dqn_policy_grid(policy_net_custom, custom_env)