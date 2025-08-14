# ==============================================================================
# 0. ライブラリのインポート
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Optional, Any

# PyTorchのインポート
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Gymnasiumのインポート
import gymnasium as gym
from gymnasium import spaces

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. Gymnasium互換の迷路環境 (MazeEnv) の定義
#    - このセクションは変更ありません
# ==============================================================================
class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
    def __init__(self, policy: np.ndarray, width: int, height: int, exit_pos: int, render_mode: Optional[str] = None):
        super().__init__()
        self.width, self.height = width, height
        self.policy, self.exit_pos = policy, exit_pos
        self.num_states, self.num_actions = self.policy.shape
        if self.num_states != self.width * self.height: raise ValueError("policyの行数が width * height と一致しません。")
        self.action_space, self.observation_space = spaces.Discrete(self.num_actions), spaces.Discrete(self.num_states)
        self.state: Optional[int] = None
        self.render_mode: Optional[str] = render_mode
        self.window_size: int = 512
        self.window = None
        self.clock = None

    def _get_obs(self) -> int: return self.state
    def _get_info(self) -> Dict[str, Any]:
        valid_actions = [i for i, v in enumerate(self.policy[self.state]) if v]
        return {"valid_actions": valid_actions}
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        return self._get_obs(), self._get_info()
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if self.state is None: raise RuntimeError("reset()を呼び出す前にstep()を呼び出すことはできません。")
        if not self.policy[self.state, action]: raise RuntimeError(f"不正な行動です: state={self.state}, action={action}")
        if action == self.UP: self.state -= self.width
        elif action == self.DOWN: self.state += self.width
        elif action == self.RIGHT: self.state += 1
        elif action == self.LEFT: self.state -= 1
        terminated = (self.state == self.exit_pos)
        reward = -0.1
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
    #def render(self): pass
    #def close(self): pass
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            gym.logger.warn("render()を呼び出していますが、render_modeが設定されていません。")
            return None
        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled("pygameがインストールされていません。`pip install pygame` を実行してください。")

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # ### 修正点 ###
        # 単一のcell_sizeではなく、幅と高さで別々のセルサイズを計算する
        cell_width = self.window_size / self.width
        cell_height = self.window_size / self.height
        line_width = max(1, int(min(cell_width, cell_height) * 0.05))

        # 壁を描画
        for state in range(self.num_states):
            y, x = divmod(state, self.width)

            
            # ### 修正点 ###
            # cell_widthとcell_heightを使って各コーナーの座標を正確に計算する
            top_left = (x * cell_width, y * cell_height)
            top_right = ((x + 1) * cell_width, y * cell_height)
            bottom_left = (x * cell_width, (y + 1) * cell_height)
            bottom_right = ((x + 1) * cell_width, (y + 1) * cell_height)


            if not self.policy[state, self.UP]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, top_right, line_width)
            if not self.policy[state, self.DOWN]:
                pygame.draw.line(canvas, (0, 0, 0), bottom_left, bottom_right, line_width)
            if not self.policy[state, self.RIGHT]:
                pygame.draw.line(canvas, (0, 0, 0), top_right, bottom_right, line_width)
            if not self.policy[state, self.LEFT]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, bottom_left, line_width)
        
        # ゴールを描画
        goal_y, goal_x = divmod(self.exit_pos, self.width)
        goal_center = (
            goal_x * cell_width + cell_width * 0.5,
            goal_y * cell_height + cell_height * 0.5,
        )
        pygame.draw.circle(
            canvas, (255, 0, 0), goal_center, min(cell_width, cell_height) * 0.3
        )

        # エージェントを描画
        if self.state is not None:
            agent_y, agent_x = divmod(self.state, self.width)
            agent_center = (
                agent_x * cell_width + cell_width * 0.5,
                agent_y * cell_height + cell_height * 0.5,
            )
            pygame.draw.circle(
                canvas, (0, 0, 255), agent_center, min(cell_width, cell_height) * 0.3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


def create_3x3_maze_policy() -> Tuple[np.ndarray, int, int]:
    width, height = 3, 3
    policy = np.zeros((width * height, 4), dtype=bool)
    allowed_moves = {0:[1,2], 1:[1,2,3], 2:[3], 3:[0,1], 4:[0,2], 5:[1,3], 6:[0,2], 7:[3], 8:[0]}
    for state, actions in allowed_moves.items(): policy[state, actions] = True
    return policy, width, height

gym.register(id='Maze-v0', entry_point='__main__:MazeEnv')

# ==============================================================================
# 2. ヘルパー関数とコア機能 (変更なし)
# ==============================================================================
def valid_actions_to_mask(valid_actions: List[int], n_actions: int) -> torch.Tensor:
    mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
    if valid_actions: mask[valid_actions] = True
    return mask

class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)
        if mask is not None: logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=-1)

def select_action_reinforce(state_tensor: torch.Tensor, policy_net: PolicyNetwork, action_mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
    probs = policy_net(state_tensor, mask=action_mask)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> torch.Tensor:
    n = len(rewards)
    returns = torch.zeros(n, device=device, dtype=torch.float32)
    discounted_return = 0.0
    for t in reversed(range(n)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return
    if standardize: returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def optimize_policy(log_probs: List[torch.Tensor], returns: torch.Tensor, optimizer: optim.Optimizer) -> float:
    loss = -torch.sum(returns * torch.stack(log_probs))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ==============================================================================
# 4. セットアップとトレーニングループ
# ==============================================================================
# Hyperparameters
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 500
MAX_STEPS_PER_EPISODE_REINFORCE = 50
STANDARDIZE_RETURNS = True
GOAL_REWARD = 10.0

# ★★★★★ 変更点 ★★★★★
# 迷路の構造情報を変数として保持しておく
maze_policy, maze_width, maze_height = create_3x3_maze_policy()
exit_position = 8

# gym.makeで環境を生成。ラッパー付きのまま `env` として使用する
env = gym.make(
    'Maze-v0',
    max_episode_steps=MAX_STEPS_PER_EPISODE_REINFORCE,
    policy=maze_policy, width=maze_width, height=maze_height, exit_pos=exit_position,
)

# 状態空間と行動空間のサイズはラッパー付きのenvからでも安全に取得可能
n_states = env.observation_space.n
n_actions = env.action_space.n
n_observations = n_states # One-Hotエンコーディングのため

print(f"Gymnasium Maze Environment (One-Hot Input, TimeLimit Wrapper Active):")
print(f"Total States (NN input dim): {n_observations}, Total Actions: {n_actions}")

policy_net_reinforce = PolicyNetwork(n_observations, n_actions).to(device)
optimizer_reinforce = optim.Adam(policy_net_reinforce.parameters(), lr=LR_REINFORCE)

episode_rewards_reinforce, episode_lengths_reinforce, episode_losses_reinforce = [], [], []

print("\nStarting REINFORCE Training...")

for i_episode in range(NUM_EPISODES_REINFORCE):
    obs, info = env.reset()
    episode_log_probs, episode_rewards = [], []
    terminated, truncated = False, False

    # トレーニングループはラッパー付きのenvをそのまま使用する
    # これにより、MAX_STEPS_PER_EPISODE_REINFORCEを超えると`truncated`がTrueになる
    while not (terminated or truncated):
        state_tensor = F.one_hot(torch.tensor(obs, device=device), num_classes=n_states).float()
        action_mask = valid_actions_to_mask(info['valid_actions'], n_actions)
        action, log_prob = select_action_reinforce(state_tensor, policy_net_reinforce, action_mask)
        episode_log_probs.append(log_prob)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated and not truncated: # ゴールした場合のみ
            reward = GOAL_REWARD

        episode_rewards.append(reward)
        obs = next_obs

    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    loss = optimize_policy(episode_log_probs, returns, optimizer_reinforce)

    episode_rewards_reinforce.append(sum(episode_rewards))
    episode_lengths_reinforce.append(len(episode_rewards))
    episode_losses_reinforce.append(loss)

    if (i_episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards_reinforce[-50:])
        avg_length = np.mean(episode_lengths_reinforce[-50:])
        print(f"Episode {i_episode+1}/{NUM_EPISODES_REINFORCE} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f}")

print("Training Finished.")

# ==============================================================================
# 5. 結果のプロット (変更なし)
# ==============================================================================
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards_reinforce)
plt.title('Episode Rewards'), plt.xlabel('Episode'), plt.ylabel('Total Reward'), plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(episode_lengths_reinforce)
plt.title('Episode Lengths'), plt.xlabel('Episode'), plt.ylabel('Steps'), plt.grid(True)
plt.tight_layout(), plt.show()


# ==============================================================================
# 6. 可視化関数の修正
# ==============================================================================
def plot_policy_on_maze(
    policy_net: PolicyNetwork,
    policy_map: np.ndarray,
    width: int,
    height: int,
    goal_pos: int,
    n_states: int,
    n_actions: int,
    device: torch.device
) -> None:
    # 行動の定義: UP = 0, DOWN = 1, RIGHT = 2, LEFT = 3
    action_symbols = {0: '↑', 1: '↓', 2: '→', 3: '←'}

    fig, ax = plt.subplots(figsize=(width * 1.2, height * 1.2))
    
    # ★★★★★ 根本的な修正 ★★★★★
    # ax.grid() は使用しない。これにより描画の競合を完全に回避する。
    # 代わりに、すべての壁を手動で描画する。
    
    # プロットの範囲とアスペクト比を設定
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # 軸の目盛りやラベルは不要なので非表示にする
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Y軸の向きを反転させて、左上を(0,0)にする
    ax.invert_yaxis()

    for s in range(n_states):
        # r: 行 (row), c: 列 (column)
        r, c = divmod(s, width)

        # policy_map に基づいて、必要な壁だけを描画する
        if not policy_map[s, 0]: # UPの壁 (セルの上辺)
            ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color='k', lw=2)
        if not policy_map[s, 1]: # DOWNの壁 (セルの下辺)
            ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color='k', lw=2)
        if not policy_map[s, 2]: # RIGHTの壁 (セルの右辺)
            ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color='k', lw=2)
        if not policy_map[s, 3]: # LEFTの壁 (セルの左辺)
            ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color='k', lw=2)

        # S, G, 方策の矢印を描画するロジックは変更なし
        if s == goal_pos:
            ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=22, weight='bold')
        elif s == 0:
            ax.text(c, r, 'S', ha='center', va='center', color='red', fontsize=18, weight='bold')
        else:
            state_tensor = F.one_hot(torch.tensor(s, device=device), num_classes=n_states).float()
            valid_actions = [i for i, v in enumerate(policy_map[s]) if v]
            if not valid_actions: continue
            action_mask = valid_actions_to_mask(valid_actions, n_actions)

            with torch.no_grad():
                action_probs = policy_net(state_tensor, mask=action_mask)
                best_action = action_probs.argmax().item()
            
            ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='blue', fontsize=20)

    ax.set_title("REINFORCE Learned Policy (Grid Corrected)", fontsize=16)
    plt.show()


print("\nPlotting Learned Policy:")
# 可視化関数の呼び出し部分は変更なし
plot_policy_on_maze(
    policy_net=policy_net_reinforce,
    policy_map=maze_policy,
    width=maze_width,
    height=maze_height,
    goal_pos=exit_position,
    n_states=n_states,
    n_actions=n_actions,
    device=device
)

env.close()