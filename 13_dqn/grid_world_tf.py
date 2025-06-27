# TensorFlowとその他の必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional

# TensorFlowをインポート
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber

# GPUが利用可能か確認し、メモリ成長を有効にする
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 現在はすべてのGPUでメモリ成長を有効にする必要がある
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using device: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # メモリ成長はプログラム開始時に設定する必要がある
        print(e)
else:
    print("Using device: CPU")


# 再現性のための乱数シードを設定
seed = 42
random.seed(seed)  # Pythonのrandomモジュール用
np.random.seed(seed)  # NumPy用
tf.random.set_seed(seed)  # TensorFlow用

# Jupyter Notebook用のインラインプロット設定
#%matplotlib inline


# カスタムグリッドワールド環境
class GridEnvironment:
    """
    シンプルな10x10のグリッドワールド環境。
    状態: (row, col)を正規化したベクトル [row/9, col/9] で表現。
    行動: 0 (上), 1 (下), 2 (左), 3 (右)。
    報酬: ゴール到達で+10, 壁に衝突で-1, 各ステップで-0.1。
    """

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        """
        Grid World環境を初期化します。

        Parameters:
        - rows (int): グリッドの行数。
        - cols (int): グリッドの列数。
        """
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)  # 開始位置
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)  # ゴール位置
        self.state: Tuple[int, int] = self.start_state  # 現在の状態
        self.state_dim: int = 2  # 状態は2つの座標 (row, col) で表現
        self.action_dim: int = 4  # 4つの離散行動: 上, 下, 左, 右

        # 行動のマッピング: 行動インデックスを(row_delta, col_delta)に変換
        self.action_map: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }

    def reset(self) -> np.ndarray:
        """
        環境を開始状態にリセットします。

        Returns:
            np.ndarray: 正規化された初期状態のNumPy配列。
        """
        self.state = self.start_state
        return self._get_state_numpy(self.state)

    def _get_state_numpy(self, state_tuple: Tuple[int, int]) -> np.ndarray:
        """
        (row, col)タプルをネットワーク用の正規化されたNumPy配列に変換します。

        Parameters:
        - state_tuple (Tuple[int, int]): タプルで表現された状態 (row, col)。

        Returns:
            np.ndarray: 正規化された状態のNumPy配列。
        """
        # 座標を0から1の間に正規化
        normalized_state: List[float] = [
            state_tuple[0] / (self.rows - 1),
            state_tuple[1] / (self.cols - 1)
        ]
        return np.array(normalized_state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        指定された行動に基づいて環境内で1ステップ進めます。

        Args:
            action (int): 実行する行動 (0:上, 1:下, 2:左, 3:右)。

        Returns:
            Tuple[np.ndarray, float, bool]: 
                - next_state_numpy (np.ndarray): 次の状態の正規化されたNumPy配列。
                - reward (float): 行動に対する報酬。
                - done (bool): エピソードが終了したかどうか。
        """
        # すでにゴールに到達している場合は、現在の状態を返す
        if self.state == self.goal_state:
            return self._get_state_numpy(self.state), 0.0, True

        # 行動に対する行と列の移動量を取得
        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        # デフォルトのステップコスト
        reward: float = -0.1
        
        # 行動が壁への衝突（範囲外）につながるかチェック
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            # 同じ状態に留まり、ペナルティを受ける
            next_row, next_col = current_row, current_col
            reward = -1.0

        # 状態を更新
        self.state = (next_row, next_col)
        next_state_numpy: np.ndarray = self._get_state_numpy(self.state)

        # ゴールに到達したかチェック
        done: bool = (self.state == self.goal_state)
        if done:
            reward = 10.0  # ゴール到達報酬

        return next_state_numpy, reward, done

    def get_action_space_size(self) -> int:
        """行動空間のサイズを返します。"""
        return self.action_dim

    def get_state_dimension(self) -> int:
        """状態表現の次元数を返します。"""
        return self.state_dim

# カスタムグリッド環境を10x10でインスタンス化
custom_env = GridEnvironment(rows=10, cols=10)

# 行動空間のサイズと状態の次元数を取得
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

# 環境に関する基本情報を表示
print(f"Custom Grid Environment:")
print(f"Size: {custom_env.rows}x{custom_env.cols}")
print(f"State Dim: {n_observations_custom}")
print(f"Action Dim: {n_actions_custom}")
print(f"Start State: {custom_env.start_state}")
print(f"Goal State: {custom_env.goal_state}")

# 環境をリセットし、開始状態の正規化された状態配列を表示
print(f"Example state numpy for (0,0): {custom_env.reset()}")

# サンプルステップ: '右' (action=3) に移動し、結果を表示
next_s, r, d = custom_env.step(3)
print(f"Step result (action=right): next_state={next_s}, reward={r}, done={d}")

# 別のサンプルステップ: '上' (action=0) に移動し、結果を表示
# エージェントは一番上の行にいるので、壁にぶつかるはず
next_s, r, d = custom_env.step(0)
print(f"Step result (action=up): next_state={next_s}, reward={r}, done={d}")


# Qネットワークのアーキテクチャを定義 (tf.keras.Modelを使用)
class DQN(Model):
    """シンプルなMLP Qネットワーク"""
    def __init__(self, n_observations: int, n_actions: int):
        """
        DQNを初期化します。

        Parameters:
        - n_observations (int): 状態空間の次元数。
        - n_actions (int): 取りうる行動の数。
        """
        super(DQN, self).__init__()
        # ネットワーク層を定義
        # シンプルなMLP: 入力 -> 隠れ層1 -> ReLU -> 隠れ層2 -> ReLU -> 出力
        self.layer1 = Dense(128, activation='relu')
        self.layer2 = Dense(128, activation='relu')
        self.layer3 = Dense(n_actions, activation=None) # Q値の出力層には活性化関数なし

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """
        ネットワークのフォワードパスを実行します。

        Parameters:
        - x (tf.Tensor): 状態を表す入力テンソル。
        - training (bool): トレーニング中かどうかを示すフラグ。

        Returns:
        - tf.Tensor: 各行動のQ値を表す出力テンソル。
        """
        # Kerasの層は入力がTensorであることを期待するが、NumPy配列も自動で変換される
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# 遷移を保存するための構造を定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# リプレイメモリクラスの定義
class ReplayMemory(object):
    """遷移を保存し、バッチをサンプリングします。"""
    def __init__(self, capacity: int):
        """
        リプレイメモリを初期化します。

        Parameters:
        - capacity (int): 保存できる遷移の最大数。
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """遷移を保存します。"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """メモリからランダムなバッチをサンプリングします。"""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """現在のメモリのサイズを返します。"""
        return len(self.memory)


# 行動選択 (Epsilon-Greedy)
def select_action_custom(state: np.ndarray,
                         policy_net: Model,
                         epsilon_start: float,
                         epsilon_end: float,
                         epsilon_decay: int,
                         n_actions: int) -> Tuple[tf.Tensor, float]:
    """
    Epsilon-Greedy戦略を用いて行動を選択します。

    Parameters:
    - state (np.ndarray): 現在の状態を表すNumPy配列。
    - policy_net (Model): Q値を推定するためのQネットワーク。
    - (その他epsilon関連のパラメータ)
    - n_actions (int): 行動の数。

    Returns:
    - Tuple[tf.Tensor, float]: 選択された行動のテンソルと現在のイプシロン値。
    """
    global steps_done_custom
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done_custom / epsilon_decay)
    steps_done_custom += 1

    if sample > epsilon_threshold:
        # 活用: Q値が最も高い行動を選択
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_batch = tf.expand_dims(state_tensor, 0) # バッチ次元を追加
        q_values = policy_net(state_batch, training=False)
        action = tf.argmax(q_values, axis=1)
        action = tf.reshape(action, (1, 1))
    else:
        # 探索: ランダムに行動を選択
        action = tf.constant([[random.randrange(n_actions)]], dtype=tf.int64)
    
    return action, epsilon_threshold

# 最適化ステップ
# @tf.functionデコレータでこの関数をコンパイルし、高速化する
@tf.function
def optimize_step(
    policy_net: Model,
    target_net: Model,
    optimizer: AdamW,
    huber_loss: Huber,
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    next_states: tf.Tensor,
    dones: tf.Tensor,
    gamma: float
) -> tf.Tensor:
    """
    TensorFlowのグラフモードで1回の最適化ステップを実行します。
    """
    with tf.GradientTape() as tape:
        # Q(s_t, a): 実際に取った行動aに対するQ値をpolicy_netから計算
        q_values = policy_net(states, training=True)
        # tf.gather_ndを使用して、各状態に対応する行動のQ値を取得
        action_indices = tf.stack([tf.range(actions.shape[0], dtype=tf.int64), tf.squeeze(actions)], axis=1)
        state_action_values = tf.gather_nd(q_values, action_indices)

        # V(s_{t+1}): 次の状態s_{t+1}の最大Q値をtarget_netから計算
        # 終了状態(done=True)の価値は0とする
        next_q_values = target_net(next_states, training=False)
        next_state_max_q = tf.reduce_max(next_q_values, axis=1)
        # doneがTrueの場所では価値を0にする
        next_state_values = tf.where(dones, 0.0, next_state_max_q)
        
        # 期待Q値 (ターゲット) を計算: R + γ * max_a' Q(s', a')
        expected_state_action_values = rewards + (gamma * next_state_values)

        # 損失を計算 (Huber損失)
        loss = huber_loss(expected_state_action_values, state_action_values)

    # 勾配を計算し、policy_netの重みを更新
    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    
    return loss


def optimize_model(memory: ReplayMemory,
                   policy_net: Model,
                   target_net: Model,
                   optimizer: AdamW,
                   huber_loss: Huber,
                   batch_size: int,
                   gamma: float) -> Optional[float]:
    """
    ポリシーネットワークの最適化を1ステップ実行します。
    """
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    # 遷移をタプルからバッチに変換
    batch = Transition(*zip(*transitions))

    # NumPy配列に変換し、Noneを処理する
    states = np.array(batch.state, dtype=np.float32)
    actions = np.array(batch.action, dtype=np.int64)
    rewards = np.array(batch.reward, dtype=np.float32).flatten()
    dones = np.array(batch.done, dtype=np.bool_).flatten()

    # 次の状態にNoneが含まれる場合、ダミーの状態（例：ゼロベクトル）で置き換える
    # 終了状態の価値は`optimize_step`内でdonesフラグによって0にされるため、ダミーの値は計算に影響しない
    next_states_list = [s if s is not None else np.zeros_like(states[0]) for s in batch.next_state]
    next_states = np.array(next_states_list, dtype=np.float32)

    # テンソルに変換
    states_tf = tf.convert_to_tensor(states)
    actions_tf = tf.convert_to_tensor(actions)
    rewards_tf = tf.convert_to_tensor(rewards)
    next_states_tf = tf.convert_to_tensor(next_states)
    dones_tf = tf.convert_to_tensor(dones)
    
    # コンパイル済みの最適化ステップを呼び出す
    loss_value = optimize_step(
        policy_net, target_net, optimizer, huber_loss,
        states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf,
        gamma
    )

    return loss_value.numpy()

# ターゲットネットワークの更新
def update_target_net(policy_net: Model, target_net: Model) -> None:
    """
    ポリシーネットワークからターゲットネットワークへ重みをコピーします。
    """
    target_net.set_weights(policy_net.get_weights())


# カスタムグリッドワールド用のハイパーパラメータ
BATCH_SIZE_CUSTOM = 128
GAMMA_CUSTOM = 0.99         # 割引率
EPS_START_CUSTOM = 1.0      # εの初期値 (完全な探索)
EPS_END_CUSTOM = 0.05       # εの最終値 (5%の探索)
EPS_DECAY_CUSTOM = 10000    # εの減衰率 (大きいほどゆっくり減衰)
LR_CUSTOM = 5e-4            # 学習率
MEMORY_CAPACITY_CUSTOM = 10000
TARGET_UPDATE_FREQ_CUSTOM = 20 # ターゲットネットワークの更新頻度
NUM_EPISODES_CUSTOM = 500      # エピソード数
MAX_STEPS_PER_EPISODE_CUSTOM = 200 # 1エピソードあたりの最大ステップ数


# 環境を再インスタンス化
custom_env = GridEnvironment(rows=10, cols=10)

# 行動空間と状態空間の次元数を取得
n_actions_custom = custom_env.get_action_space_size()
n_observations_custom = custom_env.get_state_dimension()

# ポリシーネットワークとターゲットネットワークを初期化
policy_net_custom = DQN(n_observations_custom, n_actions_custom)
target_net_custom = DQN(n_observations_custom, n_actions_custom)

# ネットワークをビルドするためにダミーデータを一度渡す
policy_net_custom(tf.random.uniform((1, n_observations_custom)))
target_net_custom(tf.random.uniform((1, n_observations_custom)))

# ターゲットネットワークにポリシーネットワークの重みをコピー
update_target_net(policy_net_custom, target_net_custom)

# オプティマイザと損失関数を初期化
# `clipvalue`で勾配クリッピングを設定
optimizer_custom = AdamW(learning_rate=LR_CUSTOM, amsgrad=True, clipvalue=100.0)
huber_loss_custom = Huber()

# リプレイメモリを初期化
memory_custom = ReplayMemory(MEMORY_CAPACITY_CUSTOM)

# プロット用のリスト
episode_rewards_custom = []
episode_lengths_custom = []
episode_epsilons_custom = []
episode_losses_custom = []


# トレーニングループ
print("Starting DQN Training on Custom Grid World...")
steps_done_custom = 0

for i_episode in range(NUM_EPISODES_CUSTOM):
    state = custom_env.reset() # 初期状態を取得 (NumPy配列)
    total_reward = 0
    current_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_CUSTOM):
        action_tensor, current_epsilon = select_action_custom(
            state, policy_net_custom, EPS_START_CUSTOM, EPS_END_CUSTOM, EPS_DECAY_CUSTOM, n_actions_custom
        )
        action = action_tensor.numpy()[0][0] # テンソルからintへ変換

        next_state, reward, done = custom_env.step(action)
        total_reward += reward
        
        # リプレイメモリに保存するためにNumPy配列/値を準備
        action_mem = action_tensor.numpy()
        reward_mem = np.array([reward], dtype=np.float32)
        done_mem = np.array([done], dtype=np.bool_)
        
        # 終了状態の場合、next_stateはNoneとして保存
        memory_next_state = next_state if not done else None
        memory_custom.push(state, action_mem, memory_next_state, reward_mem, done_mem)

        state = next_state

        # ポリシーネットワークの最適化
        loss = optimize_model(
            memory_custom, policy_net_custom, target_net_custom, optimizer_custom, huber_loss_custom, BATCH_SIZE_CUSTOM, GAMMA_CUSTOM
        )
        if loss is not None:
            current_losses.append(loss)
        
        if done:
            break

    # エピソードの統計情報を保存
    episode_rewards_custom.append(total_reward)
    episode_lengths_custom.append(t + 1)
    episode_epsilons_custom.append(current_epsilon)
    episode_losses_custom.append(np.mean(current_losses) if current_losses else 0)

    # 定期的にターゲットネットワークを更新
    if i_episode % TARGET_UPDATE_FREQ_CUSTOM == 0:
        update_target_net(policy_net_custom, target_net_custom)

    # 50エピソードごとに進捗を表示
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

print("Custom Grid World Training Finished.")


# 学習結果のプロット
plt.figure(figsize=(20, 3))

# 報酬
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_custom)
plt.title('DQN Custom Grid: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
rewards_ma_custom = np.convolve(episode_rewards_custom, np.ones(50)/50, mode='valid')
if len(rewards_ma_custom) > 0:
    plt.plot(np.arange(len(rewards_ma_custom)) + 49, rewards_ma_custom, label='50-episode MA', color='orange')
plt.legend()

# ステップ数
plt.subplot(1, 3, 2)
plt.plot(episode_lengths_custom)
plt.title('DQN Custom Grid: Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
lengths_ma_custom = np.convolve(episode_lengths_custom, np.ones(50)/50, mode='valid')
if len(lengths_ma_custom) > 0:
    plt.plot(np.arange(len(lengths_ma_custom)) + 49, lengths_ma_custom, label='50-episode MA', color='orange')
plt.legend()

# Epsilon
plt.subplot(1, 3, 3)
plt.plot(episode_epsilons_custom)
plt.title('DQN Custom Grid: Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)

plt.tight_layout()
plt.show()


# 学習済み方策の分析
def plot_dqn_policy_grid(policy_net: Model, env: GridEnvironment) -> None:
    """
    DQNから得られた貪欲方策をプロットします。

    Parameters:
    - policy_net (Model): 学習済みのQネットワーク。
    - env (GridEnvironment): カスタムグリッド環境。
    """
    rows, cols = env.rows, env.cols
    policy_grid = np.empty((rows, cols), dtype=str)
    action_symbols: Dict[int, str] = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.6))

    for r in range(rows):
        for c in range(cols):
            state_tuple = (r, c)
            if state_tuple == env.goal_state:
                policy_grid[r, c] = 'G'
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=12, weight='bold')
            else:
                state_numpy = env._get_state_numpy(state_tuple)
                state_tensor = tf.convert_to_tensor(state_numpy, dtype=tf.float32)
                state_tensor = tf.expand_dims(state_tensor, 0)
                
                q_values = policy_net(state_tensor, training=False)
                best_action = tf.argmax(q_values, axis=1)[0].numpy()
                
                policy_grid[r, c] = action_symbols[best_action]
                ax.text(c, r, policy_grid[r, c], ha='center', va='center', color='black', fontsize=12)

    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DQN Learned Policy (Custom Grid)")

    plt.show()


# 学習済みネットワークの方策をプロット
print("\nPlotting Learned Policy from DQN:")
plot_dqn_policy_grid(policy_net_custom, custom_env)
