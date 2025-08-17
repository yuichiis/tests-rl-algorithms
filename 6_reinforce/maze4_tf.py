# ==============================================================================
# 0. ライブラリのインポート
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Optional, Any

# ★変更点: TensorFlow関連のライブラリをインポート
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import tensorflow_probability as tfp

# Gymnasiumのインポート (変更なし)
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

# ★変更点: デバイス設定 (TensorFlowは通常自動でGPUを検出・使用)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 必要に応じてメモリ成長を有効にする
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using device: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Using device: CPU")

# 乱数シードの固定
#seed = 42
#random.seed(seed)
#np.random.seed(seed)
## ★変更点: TensorFlowの乱数シードを固定
#tf.random.set_seed(seed)

# ==============================================================================
# 1. Gymnasium互換の迷路環境 (MazeEnv) の定義 (変更なし)
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
        self.maze_map = maze_map
        
        self._start_pos = np.array([0, 0], dtype=np.int32)
        self._goal_pos = np.array([self.height - 1, self.width - 1], dtype=np.int32)
        self._agent_location = np.copy(self._start_pos)

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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        y, x = self._agent_location
        action_mask = self.maze_map[y, x].astype(np.int8)
        return {
            "agent_position": self._agent_location,
            "action_mask": action_mask
        }

    def _get_info(self) -> Dict[str, Any]:
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._agent_location = np.copy(self._start_pos)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        y, x = self._agent_location
        if not self.maze_map[y, x, action]:
            raise ValueError(f"Invalid action {action} at position {self._agent_location}. "
                             f"Allowed actions: {np.where(self.maze_map[y, x])[0]}")

        if action == self.UP: self._agent_location[0] -= 1
        elif action == self.DOWN: self._agent_location[0] += 1
        elif action == self.RIGHT: self._agent_location[1] += 1
        elif action == self.LEFT: self._agent_location[1] -= 1
        
        terminated = np.array_equal(self._agent_location, self._goal_pos)
        reward = -1.0
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

register(
    id='Maze-v0',
    entry_point=MazeEnv,
)

def create_3x3_maze_map() -> np.ndarray:
    height, width = 3, 3
    original_actions = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
    original_allowed_moves = {
        0:[1,2], 1:[1,2,3], 2:[3], 3:[0,1], 4:[0,2], 5:[1,3], 6:[0,2], 7:[3], 8:[0]
    }
    new_action_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
    maze_map = np.zeros((height, width, 4), dtype=bool)

    for state_id, actions in original_allowed_moves.items():
        y, x = divmod(state_id, width)
        for old_action_idx in actions:
            action_name = original_actions[old_action_idx]
            new_action_idx = new_action_map[action_name]
            maze_map[y, x, new_action_idx] = True
    return maze_map

# ==============================================================================
# 2. ニューラルネットワークとヘルパー関数
# ==============================================================================

# ★変更点: tf.keras.Modelを継承してPolicyNetworkを定義
class PolicyNetwork(Model):
    """
    座標入力を受け取り、各行動のlogitsを出力する方針ネットワーク (TensorFlow版)
    """
    def __init__(self, n_actions: int):
        super(PolicyNetwork, self).__init__()
        # PyTorchのnn.Linearはtf.keras.layers.Denseに対応
        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.layer3 = layers.Dense(n_actions) # 活性化関数はlogitsを返すため不要

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # PyTorchのforwardメソッドはcallメソッドに対応
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.layer3(x)
        return logits

# ★変更点: TensorFlow/NumPyを使って収益を計算
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> tf.Tensor:
    n = len(rewards)
    # NumPyで計算を行い、最後にTensorFlowテンソルに変換する方が効率的
    returns = np.zeros(n, dtype=np.float32)
    discounted_return = 0.0
    for t in reversed(range(n)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return
        
    returns_tensor = tf.convert_to_tensor(returns)
    if standardize:
        mean = tf.math.reduce_mean(returns_tensor)
        std = tf.math.reduce_std(returns_tensor)
        returns_tensor = (returns_tensor - mean) / (std + 1e-8)
    return returns_tensor

# ==============================================================================
# 4. セットアップとトレーニングループ
# ==============================================================================
# ハイパーパラメータ (変更なし)
GAMMA_REINFORCE = 0.99
LR_REINFORCE = 1e-3
NUM_EPISODES_REINFORCE = 600
MAX_STEPS_PER_EPISODE_REINFORCE = 50
STANDARDIZE_RETURNS = True
GOAL_REWARD = 10.0

# 環境の生成 (変更なし)
maze_map_3x3 = create_3x3_maze_map()
env = gym.make('Maze-v0', maze_map=maze_map_3x3)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE_REINFORCE)

n_observations = env.observation_space["agent_position"].shape[0]
n_actions = env.action_space.n

print(f"Modern Gymnasium Maze Environment:")
print(f"Observation dim: {n_observations}, Total Actions: {n_actions}")

# ★変更点: ネットワークとオプティマイザをTensorFlowで初期化
policy_net_reinforce = PolicyNetwork(n_actions)
optimizer_reinforce = optimizers.Adam(learning_rate=LR_REINFORCE)

# ★★★★★★★★★★★★★★★★★★★★★★★★ エラー修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★★★
# ★変更点: モデルのビルド (input_shapeを指定して重みを初期化)
policy_net_reinforce.build(input_shape=(None, n_observations))
#policy_net_reinforce(tf.zeros((1, n_observations), dtype=tf.float32))
policy_net_reinforce.summary()
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

episode_rewards_reinforce, episode_lengths_reinforce, episode_losses_reinforce = [], [], []

print("\nStarting REINFORCE Training...")

for i_episode in range(NUM_EPISODES_REINFORCE):
    obs, _ = env.reset()
    episode_state, episode_action, episode_rewards, episode_mask = [], [], [], []
    
    terminated, truncated = False, False
    while not (terminated or truncated):
        agent_pos = obs['agent_position']
        action_mask = obs['action_mask']
        
        # ★変更点: 状態とマスクをTensorFlowテンソルに変換
        state_tensor = tf.convert_to_tensor(agent_pos, dtype=tf.float32)
        # マスクはbool型の方がtf.whereで扱いやすい
        action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
        
        # --- 行動選択 ---
        # TensorFlowでは勾配計算のコンテキスト外での推論は自動的にno_grad相当になる
        # モデルはバッチ入力を想定しているため、次元を追加
        state_tensor_batch = tf.expand_dims(state_tensor, axis=0)
        logits = policy_net_reinforce(state_tensor_batch)
        
        # マスクを適用して無効な行動のlogitを負の無限大にする
        masked_logits = tf.where(~action_mask_tensor, -np.inf, logits)
        
        # TensorFlow Probabilityを使ってカテゴリカル分布を生成
        m = tfp.distributions.Categorical(logits=masked_logits)
        
        action_tensor = m.sample()[0] # バッチ次元があるので[0]で取り出す
        
        # 経験を保存
        episode_action.append(action_tensor)
        episode_state.append(state_tensor)
        episode_mask.append(action_mask_tensor)

        # ★変更点: .item() -> .numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_tensor.numpy())

        episode_rewards.append(reward)
        obs = next_obs

    # 割引収益を計算
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)

    # ★変更点: TensorFlowのGradientTapeを使ってポリシーを更新
    state_tensor = tf.stack(episode_state)
    action_tensor = tf.stack(episode_action)
    mask_tensor = tf.stack(episode_mask)

    with tf.GradientTape() as tape:
        # 収集した全状態に対するlogitsを一度に計算
        logits = policy_net_reinforce(state_tensor, training=True)
        masked_logits = tf.where(~mask_tensor, -np.inf, logits)
        
        m = tfp.distributions.Categorical(logits=masked_logits)
        log_prob = m.log_prob(action_tensor)
        
        # 損失の計算
        loss = -tf.reduce_sum(returns * log_prob)

    # 勾配を計算して適用
    grads = tape.gradient(loss, policy_net_reinforce.trainable_variables)
    optimizer_reinforce.apply_gradients(zip(grads, policy_net_reinforce.trainable_variables))
    
    # ★変更点: .item() -> .numpy()
    loss_value = loss.numpy()

    episode_rewards_reinforce.append(sum(episode_rewards))
    episode_lengths_reinforce.append(len(episode_rewards))
    episode_losses_reinforce.append(loss_value)

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
# ★変更点: TensorFlowモデルを使って方策をプロット
def plot_policy_on_maze(policy_net: Model, maze_map: np.ndarray) -> None:
    height, width = maze_map.shape[:2]
    goal_pos = (height - 1, width - 1)
    
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    fig, ax = plt.subplots(figsize=(width * 1.2, height * 1.2))
    ax.set_xlim(-0.5, width - 0.5), ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.invert_yaxis()

    for r in range(height):
        for c in range(width):
            # 壁を描画
            if not maze_map[r, c, 0]: ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color='k', lw=2)
            if not maze_map[r, c, 1]: ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color='k', lw=2)
            if not maze_map[r, c, 2]: ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color='k', lw=2)
            if not maze_map[r, c, 3]: ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color='k', lw=2)

            if (r, c) == goal_pos:
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=22, weight='bold')
            elif r == 0 and c == 0:
                ax.text(c, r, 'S', ha='center', va='center', color='red', fontsize=18, weight='bold')
            else:
                action_mask = maze_map[r, c]
                if not np.any(action_mask): continue
                
                # 状態とマスクをテンソルに変換
                state_tensor = tf.convert_to_tensor([r, c], dtype=tf.float32)
                action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
                
                # logitsを取得し、マスクを適用してからargmaxで最適な行動を選択
                # バッチ次元を追加して推論
                logits = policy_net(state_tensor[tf.newaxis, :])
                masked_logits = tf.where(~action_mask_tensor, -np.inf, logits)
                
                # argmaxで最適な行動を取得し、.numpy()で値を取り出す
                best_action = tf.argmax(masked_logits, axis=1)[0].numpy()
                
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='blue', fontsize=20)

    ax.set_title("REINFORCE Learned Policy (TensorFlow)", fontsize=16)
    plt.show()

print("\nPlotting Learned Policy:")
plot_policy_on_maze(
    policy_net=policy_net_reinforce,
    maze_map=maze_map_3x3,
)