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
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Using {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        device = "/GPU:0"
    except RuntimeError as e:
        print(e)
        device = "/CPU:0"
else:
    device = "/CPU:0"
print(f"Using device: {device}")


# Set a random seed for reproducibility across runs
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# (変更なし) Environment and other utility classes
class GridEnvironment:
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
        self.action_map: Dict[int, Tuple[int, int]] = { 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1) }
    def reset(self) -> np.ndarray:
        self.state = self.start_state
        return self._get_state_array(self.state)
    def _get_state_array(self, state_tuple: Tuple[int, int]) -> np.ndarray:
        return np.array([state_tuple[0] / (self.rows - 1), state_tuple[1] / (self.cols - 1)], dtype=np.float32)
    def _get_state_tuple_from_array(self, state_array: np.ndarray) -> Tuple[int, int]:
        row = int(round(state_array[0] * (self.rows - 1)))
        col = int(round(state_array[1] * (self.cols - 1)))
        return (row, col)
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.state == self.goal_state: return self._get_state_array(self.state), 0.0, True
        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc
        reward: float = -0.1
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or self.maze[next_row, next_col] == 1:
            next_row, next_col = current_row, current_col
            reward = -1.0
        self.state = (next_row, next_col)
        done: bool = (self.state == self.goal_state)
        if done: reward = 10.0
        return self._get_state_array(self.state), reward, done
    def get_available_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        if state is None: state = self.state
        if state == self.goal_state: return []
        available_actions = []
        current_row, current_col = state
        for action, (dr, dc) in self.action_map.items():
            next_row, next_col = current_row + dr, current_col + dc
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols and self.maze[next_row, next_col] == 0:
                available_actions.append(action)
        return available_actions
    def get_action_space_size(self) -> int: return self.action_dim
    def get_state_dimension(self) -> int: return self.state_dim

default_maze = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity: int): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size: int) -> List[Transition]: return random.sample(self.memory, batch_size)
    def __len__(self) -> int: return len(self.memory)

# (変更なし) Model creation and action selection
def create_dqn_model(n_observations: int, n_actions: int) -> tf.keras.Model:
    return Sequential([InputLayer(input_shape=(n_observations,)), Dense(128, activation='relu'), Dense(128, activation='relu'), Dense(n_actions, activation='linear')])

def select_action_custom(state: np.ndarray, policy_net: tf.keras.Model, env: GridEnvironment, epsilon_start: float, epsilon_end: float, epsilon_decay: int) -> Tuple[np.ndarray, float]:
    global steps_done_custom
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done_custom / epsilon_decay)
    steps_done_custom += 1
    current_state_tuple = env._get_state_tuple_from_array(state)
    available_actions = env.get_available_actions(current_state_tuple)
    if not available_actions: return np.array([[0]], dtype=np.int64), epsilon_threshold
    if sample > epsilon_threshold:
        q_values = policy_net(np.expand_dims(state, axis=0), training=False)[0].numpy()
        mask = np.full(env.get_action_space_size(), -np.inf)
        mask[available_actions] = 0.0
        action = np.argmax(q_values + mask).reshape(1, 1)
    else:
        action = np.array([[random.choice(available_actions)]], dtype=np.int64)
    return action, epsilon_threshold

def update_target_net(policy_net: tf.keras.Model, target_net: tf.keras.Model) -> None:
    target_net.set_weights(policy_net.get_weights())


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# <<< 変更箇所 >>> @tf.functionを再度適用し、高速化
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
@tf.function
def optimize_model(
    policy_net: tf.keras.Model,
    target_net: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    criterion: tf.keras.losses.Loss,
    gamma: float,
    state_batch: tf.Tensor,
    action_batch: tf.Tensor,
    reward_batch: tf.Tensor,
    non_final_mask: tf.Tensor,
    non_final_next_states: tf.Tensor,
    next_state_available_actions_mask: tf.Tensor
) -> tf.Tensor:
    """
    Performs one step of optimization using a compiled TensorFlow graph.
    All inputs must be tensors.
    """
    with tf.GradientTape() as tape:
        # 1. 現在の状態(state_batch)に対するQ値をpolicy_netから計算し、
        #    実際にとった行動(action_batch)に対応するQ値を取得
        q_values = policy_net(state_batch, training=True)
        state_action_values = tf.gather(q_values, action_batch, batch_dims=1)

        # 2. 次の状態の価値 V(s_{t+1}) を計算
        batch_size = tf.shape(state_batch)[0]
        next_state_values = tf.zeros(batch_size, dtype=tf.float32)
        
        # 終了状態でないものだけ価値を計算
        if tf.shape(non_final_next_states)[0] > 0:
            # target_netから次の状態のQ値を取得
            target_q_values = target_net(non_final_next_states, training=False)
            
            # 事前に計算したマスクを適用して、利用不可能なアクションのQ値を-infにする
            masked_q_values = target_q_values + next_state_available_actions_mask
            
            # 利用可能なアクションの中から最大のQ値を選択
            max_q_values = tf.reduce_max(masked_q_values, axis=1)
            
            # non_finalなインデックスに値を挿入
            indices = tf.where(non_final_mask)
            next_state_values = tf.tensor_scatter_nd_update(next_state_values, indices, max_q_values)

        # 3. TDターゲット (r + γ * max_a' Q(s', a')) を計算
        expected_state_action_values = (next_state_values * gamma) + reward_batch
        
        # 4. 損失を計算
        #    Huber lossは(y_true, y_pred)の順なので、ターゲットを先に渡す
        loss = criterion(tf.expand_dims(expected_state_action_values, 1), state_action_values)

    # 5. 勾配を計算してモデルの重みを更新
    grads = tape.gradient(loss, policy_net.trainable_variables)
    clipped_grads = [tf.clip_by_value(grad, -100.0, 100.0) for grad in grads]
    optimizer.apply_gradients(zip(clipped_grads, policy_net.trainable_variables))

    return loss

# (変更なし) Hyperparameters and setup
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

custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10, maze=default_maze)
n_actions_custom: int = custom_env.get_action_space_size()
n_observations_custom: int = custom_env.get_state_dimension()

policy_net_custom = create_dqn_model(n_observations_custom, n_actions_custom)
target_net_custom = create_dqn_model(n_observations_custom, n_actions_custom)
update_target_net(policy_net_custom, target_net_custom)

optimizer_custom = AdamW(learning_rate=LR_CUSTOM, amsgrad=True)
criterion_custom = Huber()
memory_custom: ReplayMemory = ReplayMemory(MEMORY_CAPACITY_CUSTOM)

episode_rewards_custom, episode_lengths_custom, episode_epsilons_custom, episode_losses_custom = [], [], [], []

# Training Loop
print("Starting DQN Training on Custom Maze World (TensorFlow - Compiled)...")
steps_done_custom = 0

for i_episode in range(NUM_EPISODES_CUSTOM):
    state = custom_env.reset()
    total_reward = 0
    current_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_CUSTOM):
        action_array, current_epsilon = select_action_custom(state, policy_net_custom, custom_env, EPS_START_CUSTOM, EPS_END_CUSTOM, EPS_DECAY_CUSTOM)
        action = action_array.item()
        next_state, reward, done = custom_env.step(action)
        total_reward += reward
        memory_custom.push(state, action_array, next_state if not done else None, np.array([reward], dtype=np.float32), np.array([done]))
        state = next_state

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # <<< 変更箇所 >>> 学習ステップの呼び出し方を修正
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if len(memory_custom) >= BATCH_SIZE_CUSTOM:
            transitions = memory_custom.sample(BATCH_SIZE_CUSTOM)
            batch = Transition(*zip(*transitions))

            # --- ここからデータの前処理（Python/NumPyで実行） ---
            
            # バッチをNumPy配列に変換
            state_batch_np = np.array(batch.state, dtype=np.float32)
            action_batch_np = np.array(batch.action, dtype=np.int64)
            reward_batch_np = np.array(batch.reward, dtype=np.float32).flatten()
            non_final_mask_np = np.array([s is not None for s in batch.next_state], dtype=bool)
            non_final_next_states_np = np.array([s for s in batch.next_state if s is not None], dtype=np.float32)
            
            # 次の状態で利用可能なアクションのマスクを計算
            # 利用可能: 0.0, 利用不可能: -inf
            num_actions = custom_env.get_action_space_size()
            next_actions_mask_np = np.full((non_final_next_states_np.shape[0], num_actions), -np.inf, dtype=np.float32)
            next_state_tuples = [custom_env._get_state_tuple_from_array(s) for s in non_final_next_states_np]
            for i, state_tuple in enumerate(next_state_tuples):
                available_actions = custom_env.get_available_actions(state_tuple)
                if available_actions:
                    next_actions_mask_np[i, available_actions] = 0.0

            # --- 前処理ここまで。ここからテンソルに変換してコンパイル済み関数を呼び出す ---

            loss = optimize_model(
                policy_net_custom,
                target_net_custom,
                optimizer_custom,
                criterion_custom,
                GAMMA_CUSTOM,
                tf.constant(state_batch_np),
                tf.constant(action_batch_np),
                tf.constant(reward_batch_np),
                tf.constant(non_final_mask_np, dtype=tf.bool),
                tf.constant(non_final_next_states_np),
                tf.constant(next_actions_mask_np)
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
        avg_loss = np.mean([l for l in episode_losses_custom[-50:] if l > 0]) if any(l > 0 for l in episode_losses_custom[-50:]) else 0.0
        print(f"Episode {i_episode+1}/{NUM_EPISODES_CUSTOM} | Avg Reward (last 50): {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {current_epsilon:.3f}")

print("Custom Maze World Training Finished.")

# (変更なし) Plotting and policy analysis
# ... (以降のプロットコードは変更不要です) ...
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

def plot_dqn_policy_grid(policy_net: tf.keras.Model, env: GridEnvironment) -> None:
    rows, cols = env.rows, env.cols
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.7))
    ax.imshow(env.maze.astype(float), cmap='Greys', interpolation='nearest')
    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)
            if env.maze[r, c] == 1: continue
            if state_tuple == env.goal_state:
                ax.text(c, r, 'G', ha='center', va='center', color='lime', fontsize=16, weight='bold')
                continue
            available_actions = env.get_available_actions(state_tuple)
            if not available_actions: continue
            state_array: np.ndarray = env._get_state_array(state_tuple)
            q_values: np.ndarray = policy_net(np.expand_dims(state_array, axis=0), training=False).numpy()[0]
            best_action, max_q = -1, -float('inf')
            for action in available_actions:
                if q_values[action] > max_q:
                    max_q, best_action = q_values[action], action
            if best_action != -1:
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='red', fontsize=12)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DQN Learned Policy (Maze World - TensorFlow)")
    plt.show()

print("\nPlotting Learned Policy from DQN (TensorFlow):")
plot_dqn_policy_grid(policy_net_custom, custom_env)