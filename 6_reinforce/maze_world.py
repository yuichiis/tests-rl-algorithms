# Import necessary libraries for numerical computations, plotting, and utility functions
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional, Set

# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Set up the device to use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a random seed for reproducibility across runs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Enable inline plotting for Jupyter Notebook
#%matplotlib inline

# ==============================================================================
# 1. 環境クラスの変更: 壁と行動マスク機能を追加
# ==============================================================================
class GridEnvironment:
    """
    A simple 10x10 Grid World environment with optional internal walls.
    State: (row, col) represented as normalized vector [row/10, col/10].
    Actions: 0 (up), 1 (down), 2 (left), 3 (right).
    Rewards: +10 for reaching the goal, -1 for hitting a wall, -0.1 for each step.
    """
    def __init__(self, rows: int = 10, cols: int = 10, walls: Optional[Set[Tuple[int, int]]] = None) -> None:
        """
        Initializes the Grid World environment.

        Parameters:
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.
        - walls (Optional[Set[Tuple[int, int]]]): A set of (row, col) tuples representing wall locations.
        """
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
        # --- 追加: 壁の情報を保持 ---
        self.walls: Set[Tuple[int, int]] = walls if walls is not None else set()
        if self.start_state in self.walls:
            raise ValueError("Start state cannot be a wall.")
        if self.goal_state in self.walls:
            raise ValueError("Goal state cannot be a wall.")

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
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0,
            state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        ]
        return torch.tensor(normalized_state, dtype=torch.float32, device=device)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1
        
        # --- 変更: 内部の壁(self.walls)も衝突判定に含める ---
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or (next_row, next_col) in self.walls:
            next_row, next_col = current_row, current_col
            reward = -1.0

        self.state = (next_row, next_col)
        next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)

        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0

        return next_state_tensor, reward, done

    # --- 新設: 行動マスクを生成するメソッド ---
    def get_available_actions(self) -> torch.Tensor:
        """
        現在の状態で実行可能な行動のマスクを返す (True:可能, False:不可能)
        Returns a boolean tensor mask of available actions for the current state.
        """
        # 4つの行動すべてを最初は可能(True)として初期化
        available_actions = torch.ones(self.action_dim, dtype=torch.bool, device=device)
        current_row, current_col = self.state

        # 各行動について、移動先が壁や境界外でないかチェック
        for action, (dr, dc) in self.action_map.items():
            next_row, next_col = current_row + dr, current_col + dc

            # 境界外か、壁のリストに含まれている場合
            if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or \
               (next_row, next_col) in self.walls:
                available_actions[action] = False # その行動を不可能(False)にする

        return available_actions

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

# --- 迷路の壁を定義 ---
maze_walls = {
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
    (2, 2), (2, 3), (2, 4),
    (4, 0), (4, 1), (4, 2), (4, 3),
    (6, 2), (6, 3), (6, 4),
    (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
    (3, 7), (4, 7), (5, 7), (6, 7),
}

# 壁のある迷路環境をインスタンス化
custom_env = GridEnvironment(rows=10, cols=10, walls=maze_walls)

# 環境情報の表示
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()
print(f"Custom Grid Environment with Walls:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"State Dim: {n_observations_custom}, Action Dim: {n_actions_custom}")

# 開始位置での有効な行動マスクを確認
start_state_mask = custom_env.get_available_actions()
print(f"Available actions at start (0,0): {start_state_mask.cpu().numpy()} (U, D, L, R)")


# ==============================================================================
# 2. ポリシーネットワークの変更: マスク適用機能を追加
# ==============================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # --- 変更: オプションのmask引数を追加 ---
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        action_logits = self.layer3(x)

        # --- 追加: マスクが提供された場合の処理 ---
        if mask is not None:
            # マスクがFalseの要素（無効な行動）に対応するロジットを
            # 非常に小さい値(-infに近い値)で埋める
            # これにより、softmax後の確率がほぼ0になる
            action_logits = action_logits.masked_fill(~mask, -1e9)

        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

# ==============================================================================
# 3. 行動選択関数の変更: マスクを引数として受け取る
# ==============================================================================
def select_action_reinforce(
    state: torch.Tensor,
    policy_net: PolicyNetwork,
    action_mask: Optional[torch.Tensor] = None # --- 追加: action_mask引数 ---
) -> Tuple[int, torch.Tensor]:
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    # --- 変更: ポリシーネットワークにマスクを渡す ---
    action_probs = policy_net(state, mask=action_mask)

    # 以降は変更なし
    m = Categorical(action_probs.squeeze(0))
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.item(), log_prob


# Calculating Returns (変更なし)
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> torch.Tensor:
    n_steps = len(rewards)
    returns = torch.zeros(n_steps, device=device, dtype=torch.float32)
    discounted_return = 0.0
    for t in reversed(range(n_steps)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return
    if standardize:
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + 1e-8
        returns = (returns - mean_return) / std_return
    return returns

# Optimization Step (変更なし)
def optimize_policy(
    log_probs: List[torch.Tensor],
    returns: torch.Tensor,
    optimizer: optim.Optimizer
) -> float:
    log_probs_tensor = torch.stack(log_probs)
    loss = -torch.sum(returns * log_probs_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# Hyperparameters
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 2000 # 少しエピソード数を増やして学習を安定させる
MAX_STEPS_PER_EPISODE_REINFORCE = 250
STANDARDIZE_RETURNS = True

# Initialize policy network and optimizer
policy_net_reinforce: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom).to(device)
optimizer_reinforce: optim.Adam = optim.Adam(policy_net_reinforce.parameters(), lr=LR_REINFORCE)

# Lists for storing episode statistics
episode_rewards_reinforce = []
episode_lengths_reinforce = []
episode_losses_reinforce = []

# ==============================================================================
# 4. トレーニングループの変更: マスクを取得して行動選択に利用
# ==============================================================================
print("\nStarting REINFORCE Training on Custom Maze World...")

for i_episode in range(NUM_EPISODES_REINFORCE):
    state = custom_env.reset()
    episode_log_probs: List[torch.Tensor] = []
    episode_rewards: List[float] = []

    for t in range(MAX_STEPS_PER_EPISODE_REINFORCE):
        # --- 追加: 環境から現在の状態での有効な行動マスクを取得 ---
        action_mask = custom_env.get_available_actions()
        
        # --- 変更: 行動選択関数にマスクを渡す ---
        action, log_prob = select_action_reinforce(state, policy_net_reinforce, action_mask)
        episode_log_probs.append(log_prob)
        
        next_state, reward, done = custom_env.step(action)
        episode_rewards.append(reward)
        state = next_state
        
        if done:
            break
            
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    loss = optimize_policy(episode_log_probs, returns, optimizer_reinforce)
    
    total_reward = sum(episode_rewards)
    episode_rewards_reinforce.append(total_reward)
    episode_lengths_reinforce.append(t + 1)
    episode_losses_reinforce.append(loss)

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

print("Custom Maze World Training Finished (REINFORCE).")


# Plotting results (変更なし)
plt.figure(figsize=(20, 4))
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_reinforce)
plt.title('REINFORCE Custom Maze: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
rewards_ma_reinforce = np.convolve(episode_rewards_reinforce, np.ones(100)/100, mode='valid')
if len(rewards_ma_reinforce) > 0:
    plt.plot(np.arange(len(rewards_ma_reinforce)) + 99, rewards_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(episode_lengths_reinforce)
plt.title('REINFORCE Custom Maze: Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
lengths_ma_reinforce = np.convolve(episode_lengths_reinforce, np.ones(100)/100, mode='valid')
if len(lengths_ma_reinforce) > 0:
    plt.plot(np.arange(len(lengths_ma_reinforce)) + 99, lengths_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(episode_losses_reinforce)
plt.title('REINFORCE Custom Maze: Episode Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
losses_ma_reinforce = np.convolve(episode_losses_reinforce, np.ones(100)/100, mode='valid')
if len(losses_ma_reinforce) > 0:
    plt.plot(np.arange(len(losses_ma_reinforce)) + 99, losses_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

plt.tight_layout()
plt.show()


# ==============================================================================
# 5. 可視化関数の変更: 壁の描画とマスクの利用
# ==============================================================================
def plot_reinforce_policy_grid(policy_net: PolicyNetwork, env: GridEnvironment, device: torch.device) -> None:
    rows: int = env.rows
    cols: int = env.cols
    policy_grid: np.ndarray = np.empty((rows, cols), dtype=str)
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.7))

    # --- 追加: 壁を描画するための背景グリッドを作成 ---
    grid_background = np.zeros((rows, cols))
    for r_wall, c_wall in env.walls:
        grid_background[r_wall, c_wall] = 0.5 # 壁を灰色で示す

    # --- 変更: 背景に壁を描画 ---
    ax.matshow(grid_background, cmap='Greys', alpha=0.5)

    # 方策を計算するために、一時的に環境の状態を変更するので元の状態を保存
    original_state = env.state

    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)
            if state_tuple == env.goal_state:
                policy_grid[r, c] = 'G'
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=14, weight='bold')
            # --- 追加: 壁のマスには 'W' を表示 ---
            elif state_tuple in env.walls:
                policy_grid[r, c] = 'W'
                ax.text(c, r, 'W', ha='center', va='center', color='black', fontsize=12, weight='bold')
            else:
                # --- 変更: 各マスの方策を計算する際に行動マスクを使用 ---
                # 1. マスクを取得するために、環境の状態を一時的にそのマスに設定
                env.state = state_tuple
                action_mask = env.get_available_actions()
                
                # 2. ネットワークで最も確率の高い行動を計算
                state_tensor: torch.Tensor = env._get_state_tensor(state_tuple)
                with torch.no_grad():
                    state_tensor = state_tensor.unsqueeze(0)
                    # 3. ネットワークにマスクを渡して、有効な行動の中から選択させる
                    action_probs: torch.Tensor = policy_net(state_tensor, mask=action_mask)
                    best_action: int = action_probs.argmax(dim=1).item()

                policy_grid[r, c] = action_symbols[best_action]
                ax.text(c, r, policy_grid[r, c], ha='center', va='center', color='blue', fontsize=12)

    # 環境の状態を元に戻す
    env.state = original_state

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("REINFORCE Learned Policy on Maze (with Masking)")
    plt.show()

# Plot the policy learned by the trained network
print("\nPlotting Learned Policy from REINFORCE on Maze:")
plot_reinforce_policy_grid(policy_net_reinforce, custom_env, device)

