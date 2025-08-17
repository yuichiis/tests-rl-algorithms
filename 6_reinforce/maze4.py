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
from gymnasium.envs.registration import register # ★変更点: registerをインポート

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 乱数シードの固定
#seed = 42
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. Gymnasium互換の迷路環境 (MazeEnv) の定義
# ==============================================================================
class MazeEnv(gym.Env):
    """
    モダンなGymnasiumスタイルに従った迷路環境。
    - 観測空間は座標とアクションマスクを含むDict形式。
    - 状態は(y, x)座標で管理。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    # 行動の定義をクラス変数として持つ
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

    def __init__(self, maze_map: np.ndarray, render_mode: Optional[str] = None):
        super().__init__()
        if maze_map.ndim != 3 or maze_map.shape[2] != 4:
            raise ValueError("maze_mapは (height, width, 4) の形状である必要があります。")
        
        self.height, self.width = maze_map.shape[:2]
        self.maze_map = maze_map  # 壁の情報 (各セルから各方向へ進めるか)
        
        self._start_pos = np.array([0, 0], dtype=np.int32)
        self._goal_pos = np.array([self.height - 1, self.width - 1], dtype=np.int32)
        self._agent_location = np.copy(self._start_pos)

        # === 観測空間と行動空間の定義 ===
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Dict({
            "agent_position": spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self.height - 1, self.width - 1]),
                dtype=np.int32
            ),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)
        })
        
        self.render_mode = render_mode
        # (レンダリング関連の初期化は省略)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """現在の観測（座標とアクションマスク）を生成する"""
        y, x = self._agent_location
        # 現在地から移動可能な方向がTrueになっているマスクを取得
        action_mask = self.maze_map[y, x].astype(np.int8)
        return {
            "agent_position": self._agent_location,
            "action_mask": action_mask
        }

    def _get_info(self) -> Dict[str, Any]:
        """補助情報 (今回は特にないので空)"""
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._agent_location = np.copy(self._start_pos)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # ★変更点: 不正なアクションが渡された場合に例外を発生させる
        y, x = self._agent_location
        if not self.maze_map[y, x, action]:
            raise ValueError(f"Invalid action {action} at position {self._agent_location}. "
                             f"Allowed actions: {np.where(self.maze_map[y, x])[0]}")

        if action == self.UP: self._agent_location[0] -= 1
        elif action == self.DOWN: self._agent_location[0] += 1
        elif action == self.RIGHT: self._agent_location[1] += 1
        elif action == self.LEFT: self._agent_location[1] -= 1
        
        terminated = np.array_equal(self._agent_location, self._goal_pos)
        reward = -1.0  # デフォルトの報酬
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

# ★変更点: 環境をGymnasiumに登録
register(
    id='Maze-v0',
    entry_point=MazeEnv,
)

def create_3x3_maze_map() -> np.ndarray:
    """
    3x3迷路の壁情報（移動可能か）を生成する。
    元のコードの定義を安全に変換して使用する。
    """
    height, width = 3, 3
    
    # 元のコードの行動定義: UP=0, DOWN=1, RIGHT=2, LEFT=3
    original_actions = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
    original_allowed_moves = {
        0:[1,2], 1:[1,2,3], 2:[3], 3:[0,1], 4:[0,2], 5:[1,3], 6:[0,2], 7:[3], 8:[0]
    }
    
    # 新しいコードの行動定義: UP=0, RIGHT=1, DOWN=2, LEFT=3
    # このマッピングに従って変換する
    new_action_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # (height, width, 4) の形状。4は[UP, RIGHT, DOWN, LEFT]への移動可否
    maze_map = np.zeros((height, width, 4), dtype=bool)

    for state_id, actions in original_allowed_moves.items():
        y, x = divmod(state_id, width)
        for old_action_idx in actions:
            # 古い行動名を新しい行動インデックスに変換
            action_name = original_actions[old_action_idx]
            new_action_idx = new_action_map[action_name]
            maze_map[y, x, new_action_idx] = True
            
    return maze_map

# ==============================================================================
# 2. ニューラルネットワークとヘルパー関数
# ==============================================================================

class PolicyNetwork(nn.Module):
    """
    ★変更点: マスクする前のlogitsを返すように修正
    座標入力を受け取り、各行動のlogitsを出力する方針ネットワーク
    """
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 状態を正規化すると学習が安定することがある（今回は省略）
        # x = x / torch.tensor([height, width], dtype=torch.float32, device=x.device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)
        # ★変更点: マスク処理とsoftmaxはネットワークの外で行う
        return logits

# ★変更点: select_action_reinforce関数を修正
def select_action_reinforce(state_tensor: torch.Tensor, policy_net: PolicyNetwork, action_mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    方針ネットワークからlogitsを受け取り、マスクとsoftmaxを適用して行動を選択する
    """
    logits = policy_net(state_tensor)
    
    # マスクを適用して無効な行動のlogitを負の無限大にする
    masked_logits = logits.masked_fill(~action_mask.bool(), -float('inf'))
    
    # マスクされたlogitsからカテゴリカル分布を生成
    # probsを渡す代わりにlogitsを渡すことで、log_softmaxの適用が内部で行われ、
    # 数値的に安定します。
    m = Categorical(logits=masked_logits)
    
    action = m.sample()
    return action.item(), m.log_prob(action)


# --- 以下のコア関数は変更なし ---
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
# ハイパーパラメータ
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 500
MAX_STEPS_PER_EPISODE_REINFORCE = 50
STANDARDIZE_RETURNS = True
GOAL_REWARD = 10.0

# 環境の生成
maze_map_3x3 = create_3x3_maze_map()

# ★変更点: gym.make() を使って環境を生成
env = gym.make('Maze-v0', maze_map=maze_map_3x3)
# TimeLimitラッパーで最大ステップ数を設定
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE_REINFORCE)


# 状態空間と行動空間のサイズを取得
n_observations = env.observation_space["agent_position"].shape[0] # 座標の次元数 (2)
n_actions = env.action_space.n

print(f"Modern Gymnasium Maze Environment:")
print(f"Observation dim: {n_observations}, Total Actions: {n_actions}")

# ネットワークとオプティマイザの初期化
policy_net_reinforce = PolicyNetwork(n_observations, n_actions).to(device)
optimizer_reinforce = optim.Adam(policy_net_reinforce.parameters(), lr=LR_REINFORCE)

episode_rewards_reinforce, episode_lengths_reinforce, episode_losses_reinforce = [], [], []

print("\nStarting REINFORCE Training...")

# トレーニングループ本体は、select_action_reinforceの内部実装が変わっただけなので、
# 呼び出し側のコードは変更不要です。
for i_episode in range(NUM_EPISODES_REINFORCE):
    obs, _ = env.reset()
    episode_state, episode_action, episode_rewards, episode_mask = [], [], [], []
    
    terminated, truncated = False, False
    while not (terminated or truncated):
        # 観測辞書から必要な情報を取り出す
        agent_pos = obs['agent_position']
        action_mask = obs['action_mask']
        
        # テンソルに変換
        state_tensor = torch.tensor(agent_pos, device=device, dtype=torch.float32)
        action_mask_tensor = torch.tensor(action_mask, device=device)
        
        #action, log_prob = select_action_reinforce(state_tensor, policy_net_reinforce, action_mask_tensor)
        #episode_log_probs.append(log_prob)

        
        with torch.no_grad():
            logits = policy_net_reinforce(state_tensor)
            # マスクを適用して無効な行動のlogitを負の無限大にする
            masked_logits = logits.masked_fill(~action_mask_tensor.bool(), -float('inf'))
            # マスクされたlogitsからカテゴリカル分布を生成
            # probsを渡す代わりにlogitsを渡すことで、log_softmaxの適用が内部で行われ、
            # 数値的に安定します。
            #     log_softmax(x_i) = x_i - (max(x) + log(sum(exp(x_j - max(x)))))
            m = Categorical(logits=masked_logits)
            action_tensor = m.sample()
        episode_action.append(action_tensor)
        episode_state.append(state_tensor)
        episode_mask.append(action_mask_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action_tensor.item())

        episode_rewards.append(reward)
        obs = next_obs

    with torch.no_grad():
        returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)

    # update
    state_tensor = torch.stack(episode_state)
    action_tensor = torch.stack(episode_action)
    mask_tensor = torch.stack(episode_mask)

    logits = policy_net_reinforce(state_tensor)
    masked_logits = logits.masked_fill(~mask_tensor.bool(), -float('inf'))
    m = Categorical(logits=masked_logits)
    log_prob = m.log_prob(action_tensor)

    #loss = optimize_policy(episode_log_probs, returns, optimizer_reinforce)
    loss = -torch.sum(returns * log_prob)
    optimizer_reinforce.zero_grad()
    loss.backward()
    optimizer_reinforce.step()
    loss = loss.item()


    episode_rewards_reinforce.append(sum(episode_rewards))
    episode_lengths_reinforce.append(len(episode_rewards))
    episode_losses_reinforce.append(loss)

    if (i_episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards_reinforce[-50:])
        avg_length = np.mean(episode_lengths_reinforce[-50:])
        print(f"Episode {i_episode+1}/{NUM_EPISODES_REINFORCE} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f}")

print("Training Finished.")
env.close()

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
def plot_policy_on_maze(policy_net: PolicyNetwork, maze_map: np.ndarray, device: torch.device) -> None:
    height, width = maze_map.shape[:2]
    n_actions = maze_map.shape[2]
    goal_pos = (height - 1, width - 1)
    
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'} # UP, RIGHT, DOWN, LEFT
    
    fig, ax = plt.subplots(figsize=(width * 1.2, height * 1.2))
    ax.set_xlim(-0.5, width - 0.5), ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.invert_yaxis()

    for r in range(height):
        for c in range(width):
            # 壁を描画
            if not maze_map[r, c, 0]: ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color='k', lw=2) # UP
            if not maze_map[r, c, 1]: ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color='k', lw=2) # RIGHT
            if not maze_map[r, c, 2]: ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color='k', lw=2) # DOWN
            if not maze_map[r, c, 3]: ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color='k', lw=2) # LEFT

            if (r, c) == goal_pos:
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=22, weight='bold')
            elif r == 0 and c == 0:
                ax.text(c, r, 'S', ha='center', va='center', color='red', fontsize=18, weight='bold')
            else:
                action_mask = maze_map[r, c]
                if not np.any(action_mask): continue
                
                # 状態（座標）とマスクをテンソルに変換
                state_tensor = torch.tensor([r, c], device=device, dtype=torch.float32)
                action_mask_tensor = torch.tensor(action_mask, device=device)
                
                # ★変更点: logitsを取得し、マスクを適用してからargmaxで最適な行動を選択
                with torch.no_grad():
                    logits = policy_net(state_tensor)
                    masked_logits = logits.masked_fill(~action_mask_tensor.bool(), -float('inf'))
                    # softmaxは単調増加関数のため、argmaxの結果はlogitsでも確率でも同じ
                    best_action = masked_logits.argmax().item()
                
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='blue', fontsize=20)

    ax.set_title("REINFORCE Learned Policy (Modern Env)", fontsize=16)
    plt.show()


print("\nPlotting Learned Policy:")
plot_policy_on_maze(
    policy_net=policy_net_reinforce,
    maze_map=maze_map_3x3,
    device=device
)