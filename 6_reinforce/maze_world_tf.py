# 数値計算、プロット、ユーティリティ関数に必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional

# TensorFlowをインポートし、ニューラルネットワークを構築・訓練
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# TensorFlowが利用可能なGPUを検出して使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")


# 再現性のための乱数シードを設定
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

### 変更/追加 ###
# カスタムGrid World環境に行動制限機能を追加
class GridEnvironment:
    """
    迷路機能を持つ、シンプルな10x10のGrid World環境
    """
    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
        self.state: Tuple[int, int] = self.start_state
        self.state_dim: int = 2
        self.action_dim: int = 4
        self.action_map: Dict[int, Tuple[int, int]] = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)
        }
        
        # 迷路の壁を定義 (壁の座標の集合)
        self.walls = set()
        # 縦の壁
        for r in range(1, 8):
            self.walls.add((r, 4))
        # 横の壁 (出口を作る)
        for c in range(0, 8):
            if c != 2: # (5, 2)に出口を作成
                self.walls.add((5, c))

    def reset(self) -> tf.Tensor:
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> tf.Tensor:
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0,
            state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        ]
        return tf.convert_to_tensor(normalized_state, dtype=tf.float32)

    def step(self, action: int) -> Tuple[tf.Tensor, float, bool]:
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1
        
        # 移動先がグリッドの範囲外か、壁であるかチェック
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or (next_row, next_col) in self.walls:
            # 壁に衝突した場合、状態は変わらずペナルティ
            next_row, next_col = current_row, current_col
            reward = -1.0

        self.state = (next_row, next_col)
        next_state_tensor: tf.Tensor = self._get_state_tensor(self.state)

        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0

        return next_state_tensor, reward, done

    ### 変更/追加 ###
    def get_available_actions(self) -> List[bool]:
        """現在の状態から取りうる行動のマスクを返す [上, 下, 左, 右]"""
        available_actions = [True] * self.action_dim
        current_row, current_col = self.state

        for action_idx, (dr, dc) in self.action_map.items():
            next_row, next_col = current_row + dr, current_col + dc
            # 移動先が範囲外か壁なら、その行動は不可
            if not (0 <= next_row < self.rows and 0 <= next_col < self.cols) or (next_row, next_col) in self.walls:
                available_actions[action_idx] = False
        return available_actions

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim


# カスタムGrid World環境のインスタンスを作成
custom_env = GridEnvironment(rows=10, cols=10)
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

# ポリシーネットワークのアーキテクチャを定義 (変更なし)
class PolicyNetwork(Model):
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.layer3 = layers.Dense(n_actions)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

### 変更/追加 ###
# REINFORCEのための行動選択（行動マスク対応）
def select_action_reinforce(state: tf.Tensor, policy_net: PolicyNetwork, action_mask: tf.Tensor) -> Tuple[int, tf.Tensor]:
    """ポリシーネットワークの出力と行動マスクを使って行動を選択する"""
    if len(state.shape) == 1:
        state = tf.expand_dims(state, 0)
    
    action_logits = policy_net(state)

    # 不可能な行動のロジットを非常に小さい値に設定してマスク
    masked_logits = tf.where(action_mask, action_logits, -1e9)
    
    # マスクされたロジットから行動をサンプリング
    action_tensor = tf.random.categorical(masked_logits, 1)
    action = tf.squeeze(action_tensor, axis=-1).numpy()[0]
    
    # マスクされたロジットから対数確率を計算
    log_softmax_logits = tf.nn.log_softmax(masked_logits)
    log_prob = tf.gather_nd(log_softmax_logits, indices=[[0, action]])
    
    return action, log_prob

# 収益の計算 (変更なし)
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> tf.Tensor:
    n_steps = len(rewards)
    returns_np = np.zeros(n_steps, dtype=np.float32)
    discounted_return = 0.0
    for t in reversed(range(n_steps)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns_np[t] = discounted_return
    returns = tf.convert_to_tensor(returns_np)
    if standardize:
        mean_return = tf.reduce_mean(returns)
        std_return = tf.math.reduce_std(returns) + 1e-8
        returns = (returns - mean_return) / std_return
    return returns

### 変更/追加 ###
# 最適化ステップ（行動マスク対応）
def optimize_policy(
    episode_states: List[tf.Tensor],
    episode_actions: List[int],
    episode_masks: List[tf.Tensor], # マスクのリストを追加
    returns: tf.Tensor,
    policy_net: PolicyNetwork,
    optimizer: Adam
) -> float:
    """tf.GradientTapeを使用してポリシーネットワークの最適化を1ステップ実行する"""
    states_tensor = tf.stack(episode_states)
    actions_tensor = tf.convert_to_tensor(episode_actions, dtype=tf.int32)
    masks_tensor = tf.stack(episode_masks) # マスクをテンソルに変換

    with tf.GradientTape() as tape:
        action_logits = policy_net(states_tensor, training=True)
        
        # 記録されたマスクを使ってロジットをマスキング
        masked_logits = tf.where(masks_tensor, action_logits, -1e9)
        
        # マスクされたロジットから対数確率を計算
        log_softmax_logits = tf.nn.log_softmax(masked_logits)
        indices = tf.stack([tf.range(len(actions_tensor)), actions_tensor], axis=1)
        log_probs = tf.gather_nd(log_softmax_logits, indices)
        
        loss = -tf.reduce_sum(returns * log_probs)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    return loss.numpy()


# ハイパーパラメータ (エピソード数を少し増やす)
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 2500 # 迷路が複雑になったためエピソード数を増やす
MAX_STEPS_PER_EPISODE_REINFORCE = 250
STANDARDIZE_RETURNS = True

# 環境、モデル、オプティマイザの初期化
policy_net_reinforce: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom)
optimizer_reinforce: Adam = Adam(learning_rate=LR_REINFORCE)
episode_rewards_reinforce = []
episode_lengths_reinforce = []
episode_losses_reinforce = []

# トレーニング
print("Starting REINFORCE Training on Custom Grid World with Maze...")

for i_episode in range(NUM_EPISODES_REINFORCE):
    state = custom_env.reset()
    
    episode_states: List[tf.Tensor] = []
    episode_actions: List[int] = []
    episode_rewards: List[float] = []
    episode_masks: List[tf.Tensor] = [] ### 変更/追加 ###
    
    for t in range(MAX_STEPS_PER_EPISODE_REINFORCE):
        episode_states.append(state)
        
        ### 変更/追加 ###
        # 現在の状態で可能な行動マスクを取得
        action_mask = custom_env.get_available_actions()
        action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
        episode_masks.append(action_mask_tensor)
        
        # マスクを使って行動を選択
        action, _ = select_action_reinforce(state, policy_net_reinforce, action_mask_tensor)
        
        episode_actions.append(action)
        next_state, reward, done = custom_env.step(action)
        episode_rewards.append(reward)
        
        state = next_state
        if done:
            break
            
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    
    ### 変更/追加 ###
    # マスクを渡してポリシーを更新
    loss = optimize_policy(
        episode_states, episode_actions, episode_masks, returns, policy_net_reinforce, optimizer_reinforce
    )
    
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

print("Custom Grid World Training Finished (REINFORCE).")


# プロット部分は変更なし
plt.figure(figsize=(20, 4))
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_reinforce)
plt.title('REINFORCE Custom Grid: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
rewards_ma_reinforce = np.convolve(episode_rewards_reinforce, np.ones(100)/100, mode='valid')
if len(rewards_ma_reinforce) > 0: 
    plt.plot(np.arange(len(rewards_ma_reinforce)) + 99, rewards_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()
# (以下、他のプロットも同様のため省略)
plt.subplot(1, 3, 2)
plt.plot(episode_lengths_reinforce)
plt.title('REINFORCE Custom Grid: Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
lengths_ma_reinforce = np.convolve(episode_lengths_reinforce, np.ones(100)/100, mode='valid')
if len(lengths_ma_reinforce) > 0:
    plt.plot(np.arange(len(lengths_ma_reinforce)) + 99, lengths_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(episode_losses_reinforce)
plt.title('REINFORCE Custom Grid: Episode Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
losses_ma_reinforce = np.convolve(episode_losses_reinforce, np.ones(100)/100, mode='valid')
if len(losses_ma_reinforce) > 0:
    plt.plot(np.arange(len(losses_ma_reinforce)) + 99, losses_ma_reinforce, label='100-episode MA', color='orange')
plt.legend()

plt.tight_layout()
plt.show()


### 変更/追加 ###
# 学習済みポリシーの分析（壁の描画機能を追加）
def plot_reinforce_policy_grid(policy_net: PolicyNetwork, env: GridEnvironment) -> None:
    """学習済みポリシーと迷路の壁をプロットする"""
    rows: int = env.rows
    cols: int = env.cols
    policy_grid: np.ndarray = np.empty((rows, cols), dtype=str)
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.8, rows * 0.8))

    # 壁を描画
    wall_patch = np.zeros((rows, cols))
    for r_wall, c_wall in env.walls:
        wall_patch[r_wall, c_wall] = 1
    ax.imshow(wall_patch, cmap='Greys', interpolation='nearest')


    for r in range(rows):
        for c in range(cols):
            state_tuple: Tuple[int, int] = (r, c)
            if state_tuple in env.walls:
                ax.text(c, r, '■', ha='center', va='center', color='black', fontsize=12)
            elif state_tuple == env.goal_state:
                ax.text(c, r, 'G', ha='center', va='center', color='red', fontsize=12, weight='bold')
            else:
                state_tensor: tf.Tensor = env._get_state_tensor(state_tuple)
                
                # 行動マスクを取得して適用
                env.state = state_tuple # 一時的に状態を設定してマスクを取得
                action_mask = env.get_available_actions()
                action_mask_tensor = tf.convert_to_tensor([action_mask], dtype=tf.bool)
                
                # マスクを適用して最も確率の高い行動を選択
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_logits: tf.Tensor = policy_net(state_tensor)
                masked_logits = tf.where(action_mask_tensor, action_logits, -1e9)
                
                best_action: int = tf.argmax(masked_logits, axis=1).numpy()[0]
                
                policy_grid[r, c] = action_symbols[best_action]
                ax.text(c, r, policy_grid[r, c], ha='center', va='center', color='blue', fontsize=10)

    # グリッドの描画設定
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("REINFORCE Learned Policy (with Maze)")
    plt.show()


print("\nPlotting Learned Policy from REINFORCE:")
plot_reinforce_policy_grid(policy_net_reinforce, custom_env)