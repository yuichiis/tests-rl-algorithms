# Import necessary libraries for numerical computations, plotting, and utility functions
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional

# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


# <<< 変更 >>> Custom Grid World Environmentを迷路対応に修正
class GridEnvironment:
    """
    A simple 10x10 Grid World environment with walls (a maze).
    State: (row, col) represented as normalized vector [row/10, col/10].
    Actions: 0 (up), 1 (down), 2 (left), 3 (right).
    Rewards: +10 for reaching the goal, -1 for hitting a wall, -0.1 for each step.
    Action selection is restricted by walls.
    """

    def __init__(self, rows: int = 10, cols: int = 10, maze: Optional[np.ndarray] = None) -> None:
        """
        Initializes the Grid World environment.

        Parameters:
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.
        - maze (np.ndarray, optional): A 2D numpy array representing the maze.
                                        0 for path, 1 for wall. If None, an open grid is created.
        """
        self.rows: int = rows
        self.cols: int = cols

        # <<< 追加 >>> 迷路の定義
        if maze is None:
            # 迷路が指定されない場合は、壁のないオープンなグリッドを作成
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

    def reset(self) -> torch.Tensor:
        """
        Resets the environment to the start state.
        """
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
        """
        Converts a (row, col) tuple to a normalized tensor for the network.
        """
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1),
            state_tuple[1] / (self.cols - 1)
        ]
        return torch.tensor(normalized_state, dtype=torch.float32, device=device)

    # <<< 追加 >>> テンソルから状態タプルへの変換（ヘルパー関数）
    def _get_state_tuple_from_tensor(self, state_tensor: torch.Tensor) -> Tuple[int, int]:
        """Converts a normalized state tensor back to a (row, col) tuple."""
        row = int(round(state_tensor[0].item() * (self.rows - 1)))
        col = int(round(state_tensor[1].item() * (self.cols - 1)))
        return (row, col)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Performs one step in the environment based on the given action.
        """
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1  # Default step cost

        # <<< 変更 >>> 壁のチェックを追加
        # 境界外または壁にぶつかったかチェック
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or self.maze[next_row, next_col] == 1:
            # 状態は変更せず、壁衝突のペナルティを与える
            next_row, next_col = current_row, current_col
            reward = -1.0
        
        self.state = (next_row, next_col)
        next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)

        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0

        return next_state_tensor, reward, done

    # <<< 追加 >>> 利用可能な行動を取得するメソッド
    def get_available_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """
        Returns a list of available actions for a given state, considering walls.
        
        Parameters:
        - state (Tuple[int, int], optional): The state (row, col). If None, uses self.state.

        Returns:
        - List[int]: A list of action indices that are possible from the given state.
        """
        if state is None:
            state = self.state
        
        # ゴールに到達している場合は行動不能
        if state == self.goal_state:
            return []

        available_actions = []
        current_row, current_col = state

        for action, (dr, dc) in self.action_map.items():
            next_row, next_col = current_row + dr, current_col + dc
            # 境界内で、かつ壁でない場合のみ行動可能
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
print(f"Available actions at start state (0,0): {custom_env.get_available_actions((0,0))}") # Should be [1, 3] (Down, Right)
print(f"Available actions at (2,1): {custom_env.get_available_actions((2,1))}") # Should be [1, 2, 3] (Down, Left, Right)


# Define the Q-Network architecture (変更なし)
class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

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


# <<< 変更 >>> Action Selectionを迷路対応に修正
def select_action_custom(state: torch.Tensor,
                         policy_net: nn.Module,
                         env: GridEnvironment, # <<< 追加
                         epsilon_start: float,
                         epsilon_end: float,
                         epsilon_decay: int) -> Tuple[torch.Tensor, float]:
    """
    Selects an action using epsilon-greedy, restricted to available actions.
    """
    global steps_done_custom
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done_custom / epsilon_decay)
    steps_done_custom += 1

    # <<< 追加 >>> 現在の状態で利用可能な行動を取得
    current_state_tuple = env._get_state_tuple_from_tensor(state)
    available_actions = env.get_available_actions(current_state_tuple)

    # 利用可能な行動がない場合は、ダミーの行動を返す（通常はゴールに到達した場合）
    if not available_actions:
        return torch.tensor([[0]], device=device, dtype=torch.long), epsilon_threshold

    if sample > epsilon_threshold:
        # Exploitation: 利用可能な行動の中で最善の行動を選択
        with torch.no_grad():
            state_batch = state.unsqueeze(0)
            q_values = policy_net(state_batch)[0]
            
            # 利用不可能な行動のQ値を非常に低い値にマスク
            mask = torch.full((env.get_action_space_size(),), -float('inf'), device=device)
            mask[available_actions] = 0.0
            masked_q_values = q_values + mask

            # マスクされたQ値から最大値を持つ行動を選択
            action = masked_q_values.argmax().view(1, 1)
    else:
        # Exploration: 利用可能な行動の中からランダムに選択
        action = torch.tensor([[random.choice(available_actions)]], device=device, dtype=torch.long)

    return action, epsilon_threshold


# <<< 変更 >>> Optimization Stepを迷路対応に修正
def optimize_model(memory: ReplayMemory,
                   policy_net: nn.Module,
                   target_net: nn.Module,
                   optimizer: optim.Optimizer,
                   env: GridEnvironment, # <<< 追加
                   batch_size: int,
                   gamma: float,
                   criterion: nn.Module = nn.SmoothL1Loss()) -> Optional[float]:
    """
    Performs one step of optimization, considering available actions for the next state.
    """
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    # <<< 変更 >>> 次の状態で利用可能な行動を考慮してTDターゲットを計算
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            # ターゲットネットワークから次の状態のQ値を取得
            target_q_values = target_net(non_final_next_states)
            
            # 各次の状態について、利用可能な行動を取得
            next_state_tuples = [env._get_state_tuple_from_tensor(s) for s in non_final_next_states]
            
            # 利用不可能な行動のQ値をマスク
            mask = torch.full_like(target_q_values, -float('inf'))
            for i, state_tuple in enumerate(next_state_tuples):
                available_actions = env.get_available_actions(state_tuple)
                if available_actions:
                    mask[i, available_actions] = 0.0
            
            # マスクされたQ値から各状態の最大値を取得
            max_q_values = (target_q_values + mask).max(1)[0]
            next_state_values[non_final_mask] = max_q_values

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


# Target Network Update (変更なし)
def update_target_net(policy_net: nn.Module, target_net: nn.Module) -> None:
    target_net.load_state_dict(policy_net.state_dict())


# <<< 変更 >>> ハイパーパラメータを迷路用に調整
BATCH_SIZE_CUSTOM = 128
GAMMA_CUSTOM = 0.99
EPS_START_CUSTOM = 1.0
EPS_END_CUSTOM = 0.05
EPS_DECAY_CUSTOM = 20000  # 探索期間を長くする
TAU_CUSTOM = 0.005
LR_CUSTOM = 5e-4
MEMORY_CAPACITY_CUSTOM = 20000 # メモリ容量を増やす
TARGET_UPDATE_FREQ_CUSTOM = 20
NUM_EPISODES_CUSTOM = 1000 # エピソード数を増やす
MAX_STEPS_PER_EPISODE_CUSTOM = 500 # 1エピソードの最大ステップ数を増やす

# Re-instantiate the maze environment
custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10, maze=default_maze)

n_actions_custom: int = custom_env.get_action_space_size()
n_observations_custom: int = custom_env.get_state_dimension()

policy_net_custom: DQN = DQN(n_observations_custom, n_actions_custom).to(device)
target_net_custom: DQN = DQN(n_observations_custom, n_actions_custom).to(device)
target_net_custom.load_state_dict(policy_net_custom.state_dict())
target_net_custom.eval()

optimizer_custom: optim.AdamW = optim.AdamW(policy_net_custom.parameters(), lr=LR_CUSTOM, amsgrad=True)
memory_custom: ReplayMemory = ReplayMemory(MEMORY_CAPACITY_CUSTOM)

episode_rewards_custom = []
episode_lengths_custom = []
episode_epsilons_custom = []
episode_losses_custom = []


# Training Loop
print("Starting DQN Training on Custom Maze World...")
steps_done_custom = 0

for i_episode in range(NUM_EPISODES_CUSTOM):
    state = custom_env.reset()
    total_reward = 0
    current_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_CUSTOM):
        # <<< 変更 >>> select_action_customの呼び出し
        action_tensor, current_epsilon = select_action_custom(
            state, policy_net_custom, custom_env, EPS_START_CUSTOM, EPS_END_CUSTOM, EPS_DECAY_CUSTOM
        )
        action = action_tensor.item()

        next_state_tensor, reward, done = custom_env.step(action)
        total_reward += reward

        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        action_tensor_mem = torch.tensor([[action]], device=device, dtype=torch.long)
        done_tensor = torch.tensor([done], device=device, dtype=torch.bool)

        memory_next_state = next_state_tensor if not done else None
        memory_custom.push(state, action_tensor_mem, memory_next_state, reward_tensor, done_tensor)

        state = next_state_tensor

        # <<< 変更 >>> optimize_modelの呼び出し
        loss = optimize_model(
            memory_custom, policy_net_custom, target_net_custom, optimizer_custom, custom_env, BATCH_SIZE_CUSTOM, GAMMA_CUSTOM
        )
        if loss is not None:
            current_losses.append(loss)

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


# <<< 変更 >>> Analyzing the Learned Policy (迷路対応)
def plot_dqn_policy_grid(policy_net: nn.Module, env: GridEnvironment, device: torch.device) -> None:
    """
    Plots the greedy policy derived from the DQN, including maze walls.
    """
    rows: int = env.rows
    cols: int = env.cols
    policy_grid: np.ndarray = np.empty((rows, cols), dtype=str)
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.7))

    # <<< 追加 >>> 壁を描画
    # 0:道(白), 1:壁(黒)のカラーマップを作成
    maze_display = env.maze.astype(float)
    ax.imshow(maze_display, cmap='Greys', interpolation='nearest')

    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)

            # 壁やゴールには方策を描画しない
            if env.maze[r, c] == 1:
                continue
            if state_tuple == env.goal_state:
                ax.text(c, r, 'G', ha='center', va='center', color='lime', fontsize=16, weight='bold')
                continue
            
            # <<< 変更 >>> 利用可能な行動を考慮して最善の行動を決定
            available_actions = env.get_available_actions(state_tuple)
            if not available_actions:
                continue # 行動不能な場所（袋小路など）

            state_tensor: torch.Tensor = env._get_state_tensor(state_tuple)
            with torch.no_grad():
                state_tensor = state_tensor.unsqueeze(0)
                q_values: torch.Tensor = policy_net(state_tensor)[0]
                
                # 利用可能な行動の中から最善手を選択
                best_action = -1
                max_q = -float('inf')
                for action in available_actions:
                    if q_values[action] > max_q:
                        max_q = q_values[action]
                        best_action = action
                
                # 方策をグリッドに描画
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='red', fontsize=12)

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DQN Learned Policy (Maze World)")
    plt.show()

# Plot the policy learned by the trained network
print("\nPlotting Learned Policy from DQN:")
plot_dqn_policy_grid(policy_net_custom, custom_env, device)