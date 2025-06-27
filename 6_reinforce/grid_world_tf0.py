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
        # 現在はメモリ成長を有効にする必要がある
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # プログラム開始時にメモリ成長を設定する必要がある
        print(e)
else:
    print("No GPU found, using CPU.")


# 再現性のための乱数シードを設定
seed = 42
random.seed(seed)  # Pythonのrandomモジュールのシード
np.random.seed(seed)  # NumPyのシード
tf.random.set_seed(seed)  # TensorFlowのシード

# Jupyter Notebook用のインラインプロットを有効化
#%matplotlib inline

# カスタムGrid World環境 (PyTorch版と同一)
class GridEnvironment:
    """
    シンプルな10x10のGrid World環境
    状態: (行, 列) を正規化されたベクトル [row/9, col/9] で表現
    行動: 0 (上), 1 (下), 2 (左), 3 (右)
    報酬: ゴール到達で+10, 壁に衝突で-1, 1ステップごとに-0.1
    """

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
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

    def reset(self) -> tf.Tensor:
        """環境を初期状態にリセットする"""
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> tf.Tensor:
        """(行, 列)タプルを正規化されたテンソルに変換する"""
        # 座標を0と1の間に正規化
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0,
            state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        ]
        # TensorFlowのテンソルに変換
        return tf.convert_to_tensor(normalized_state, dtype=tf.float32)

    def step(self, action: int) -> Tuple[tf.Tensor, float, bool]:
        """与えられた行動に基づいて環境内で1ステップ進める"""
        if self.state == self.goal_state:
            return self._get_state_tensor(self.state), 0.0, True

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1
        
        # 壁に衝突したか（範囲外か）をチェック
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            next_row, next_col = current_row, current_col
            reward = -1.0

        self.state = (next_row, next_col)
        next_state_tensor: tf.Tensor = self._get_state_tensor(self.state)

        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0

        return next_state_tensor, reward, done

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

# カスタムGrid World環境のインスタンスを作成
custom_env = GridEnvironment(rows=10, cols=10)
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

print(f"Custom Grid Environment:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"State Dim: {n_observations_custom}")
print(f"Action Dim: {n_actions_custom}")
print(f"Start State: {custom_env.start_state}")
print(f"Goal State: {custom_env.goal_state}")
print(f"Example state tensor for (0,0): {custom_env.reset()}")
next_s, r, d = custom_env.step(3)
print(f"Step result (action=right): next_state={next_s.numpy()}, reward={r}, done={d}")
custom_env.reset()
next_s, r, d = custom_env.step(0)
print(f"Step result (action=up): next_state={next_s.numpy()}, reward={r}, done={d}")


# ポリシーネットワークのアーキテクチャを定義
class PolicyNetwork(Model):
    """ REINFORCE用のシンプルなMLPポリシーネットワーク """
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        # Kerasのレイヤーを使用してネットワークを定義
        self.layer1 = layers.Dense(128, activation='relu', name='policy_layer1')
        self.layer2 = layers.Dense(128, activation='relu', name='policy_layer2')
        # 出力層はSoftmaxを適用する前の「ロジット」を返す
        self.layer3 = layers.Dense(n_actions, name='policy_output_logits')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        ネットワークを順伝播し、行動のロジットを取得する
        
        Parameters:
        - x (tf.Tensor): 状態を表す入力テンソル
        Returns:
        - tf.Tensor: 行動のロジットを表す出力テンソル
        """
        x = tf.cast(x, tf.float32) # 入力をfloat32に変換
        x = self.layer1(x)
        x = self.layer2(x)
        action_logits = self.layer3(x)
        return action_logits

# REINFORCEのための行動選択
def select_action_reinforce(state: tf.Tensor, policy_net: PolicyNetwork) -> Tuple[int, tf.Tensor]:
    """
    ポリシーネットワークの出力分布からサンプリングして行動を選択する

    Returns:
    - Tuple[int, tf.Tensor]:
        - action (int): 選択された行動のインデックス
        - log_prob (tf.Tensor): 選択された行動の対数確率
    """
    # 状態の次元が1の場合、バッチ次元を追加 [state_dim] -> [1, state_dim]
    if len(state.shape) == 1:
        state = tf.expand_dims(state, 0)

    # ネットワークから行動のロジットを取得
    action_logits = policy_net(state)

    # ロジットから行動をサンプリング
    # tf.random.categoricalはロジットを入力として受け取り、サンプリング結果を返す
    action_tensor = tf.random.categorical(action_logits, 1)
    action = tf.squeeze(action_tensor, axis=-1).numpy()[0]

    # 選択された行動の対数確率を計算
    log_softmax_logits = tf.nn.log_softmax(action_logits)
    # tf.gather_ndを使用して、選択された行動に対応する対数確率を取得
    log_prob = tf.gather_nd(log_softmax_logits, indices=[[0, action]])

    return action, log_prob

# 収益の計算
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> tf.Tensor:
    """
    エピソードの各ステップtにおける割引収益G_tを計算する
    """
    n_steps = len(rewards)
    # NumPy配列で計算し、最後にTensorFlowテンソルに変換する方が簡単
    returns_np = np.zeros(n_steps, dtype=np.float32)
    discounted_return = 0.0

    # 報酬を逆順にループして割引収益を計算
    for t in reversed(range(n_steps)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns_np[t] = discounted_return

    returns = tf.convert_to_tensor(returns_np)

    # 収益を標準化（平均を引き、標準偏差で割る）
    if standardize:
        mean_return = tf.reduce_mean(returns)
        std_return = tf.math.reduce_std(returns) + 1e-8 # ゼロ除算を防止
        returns = (returns - mean_return) / std_return

    return returns

# 最適化ステップ
def optimize_policy(
    log_probs: List[tf.Tensor], 
    returns: tf.Tensor, 
    policy_net: PolicyNetwork,
    optimizer: Adam
) -> float:
    """
    tf.GradientTapeを使用してポリシーネットワークの最適化を1ステップ実行する
    """
    with tf.GradientTape() as tape:
        # log_probsのリストを一つのテンソルにまとめる
        # 各log_probの形状は(1,)なので、squeezeしてスカラーにしてからstackする
        log_probs_tensor = tf.stack([tf.squeeze(lp) for lp in log_probs])
        
        # REINFORCEの損失を計算: - (収益 * 対数確率)
        # E[G_t * log(pi)]を最大化するため、-E[G_t * log(pi)]を最小化する
        loss = -tf.reduce_sum(returns * log_probs_tensor)

    # 勾配を計算
    grads = tape.gradient(loss, policy_net.trainable_variables)
    # 勾配を適用してネットワークのパラメータを更新
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

    return loss.numpy() # ログ記録のために損失値を返す

# REINFORCEのハイパーパラメータ
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 1500
MAX_STEPS_PER_EPISODE_REINFORCE = 200
STANDARDIZE_RETURNS = True

# カスタムGridEnvironmentを再インスタンス化
custom_env: GridEnvironment = GridEnvironment(rows=10, cols=10)
n_actions_custom: int = custom_env.get_action_space_size()
n_observations_custom: int = custom_env.get_state_dimension()

# ポリシーネットワークを初期化
policy_net_reinforce: PolicyNetwork = PolicyNetwork(n_observations_custom, n_actions_custom)
# オプティマイザを初期化
optimizer_reinforce: Adam = Adam(learning_rate=LR_REINFORCE)

# プロット用の統計情報を格納するリスト
episode_rewards_reinforce = []
episode_lengths_reinforce = []
episode_losses_reinforce = []

# トレーニング
print("Starting REINFORCE Training on Custom Grid World...")

# トレーニングループ
for i_episode in range(NUM_EPISODES_REINFORCE):
    state = custom_env.reset()
    
    episode_log_probs: List[tf.Tensor] = []
    episode_rewards: List[float] = []
    
    # --- 1エピソードを生成 ---
    for t in range(MAX_STEPS_PER_EPISODE_REINFORCE):
        action, log_prob = select_action_reinforce(state, policy_net_reinforce)
        episode_log_probs.append(log_prob)
        
        next_state, reward, done = custom_env.step(action)
        episode_rewards.append(reward)
        
        state = next_state
        
        if done:
            break
            
    # --- エピソード終了後、ポリシーを更新 ---
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    loss = optimize_policy(episode_log_probs, returns, policy_net_reinforce, optimizer_reinforce)
    
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


# REINFORCEの結果をプロット
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

# 学習済みポリシーの分析
def plot_reinforce_policy_grid(policy_net: PolicyNetwork, env: GridEnvironment) -> None:
    """
    REINFORCEポリシーネットワークから導出された貪欲方策をプロットする
    （サンプリングではなく、最も確率の高い行動を表示）
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
                state_tensor: tf.Tensor = env._get_state_tensor(state_tuple)
                # 勾配計算は不要
                state_tensor = tf.expand_dims(state_tensor, 0)
                # 行動のロジットを取得
                action_logits: tf.Tensor = policy_net(state_tensor)
                # 最も確率の高い行動（貪欲行動）を選択
                best_action: int = tf.argmax(action_logits, axis=1).numpy()[0]

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

# 学習済みネットワークのポリシーをプロット
print("\nPlotting Learned Policy from REINFORCE:")
plot_reinforce_policy_grid(policy_net_reinforce, custom_env)
