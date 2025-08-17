# ==============================================================================
# 0. ライブラリのインポート
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Optional, Any

# TensorFlow関連のライブラリをインポート
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# Gymnasiumのインポート
import gymnasium as gym
from gymnasium import spaces

# デバイスの設定 (TensorFlow版)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    device_name = "GPU"
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    device_name = "CPU"
print(f"Using device: {device_name}")

# 乱数シードの設定
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ==============================================================================
# 1. Gymnasium互換の迷路環境 (MazeEnv) の定義 (変更なし)
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
    def render(self): pass
    def close(self): pass

def create_3x3_maze_policy() -> Tuple[np.ndarray, int, int]:
    width, height = 3, 3
    policy = np.zeros((width * height, 4), dtype=bool)
    allowed_moves = {0:[1,2], 1:[1,2,3], 2:[3], 3:[0,1], 4:[0,2], 5:[1,3], 6:[0,2], 7:[3], 8:[0]}
    for state, actions in allowed_moves.items(): policy[state, actions] = True
    return policy, width, height

# 登録時に `__main__` を指定しているため、スクリプトとして実行する必要がある
gym.register(id='Maze-v0', entry_point='__main__:MazeEnv')

# ==============================================================================
# 2. ヘルパー関数とコア機能 (TensorFlow版)
# ==============================================================================
def valid_actions_to_mask(valid_actions: List[int], n_actions: int) -> tf.Tensor:
    """有効な行動のリストからTensorFlowのブールマスクを作成"""
    if not valid_actions:
        return tf.zeros(n_actions, dtype=tf.bool)
    mask = tf.reduce_sum(tf.one_hot(valid_actions, depth=n_actions), axis=0)
    return tf.cast(mask, tf.bool)

class PolicyNetwork(keras.Model):
    """KerasのModelを継承した方策ネットワーク"""
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.layer3 = layers.Dense(n_actions) # Softmaxは適用しない

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """softmaxを適用する前のロジットを返す"""
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.layer3(x)
        if mask is not None:
            mask_float = tf.cast(mask, dtype=tf.float32)
            if len(mask_float.shape) < len(logits.shape):
                mask_float = tf.expand_dims(mask_float, 0)
            logits += (1.0 - mask_float) * -1e9
        return logits

def select_action_reinforce(state_tensor: tf.Tensor, policy_net: PolicyNetwork, action_mask: tf.Tensor) -> Tuple[int, tf.Tensor]:
    """方策に基づいて行動を選択し、その対数確率を返す"""
    if len(state_tensor.shape) == 1:
        state_tensor = tf.expand_dims(state_tensor, 0)

    # モデルからlogitsを取得
    logits = policy_net(state_tensor, mask=action_mask)
    # TensorFlow ProbabilityのCategorical分布にlogitsを直接渡す
    m = tfp.distributions.Categorical(logits=logits)
    action = m.sample()
    # log_probは最適化では使わないが、デバッグ等のために返す
    log_prob = m.log_prob(action)
    return action.numpy()[0], log_prob[0]

def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> tf.Tensor:
    """報酬のリストから割引収益を計算し、任意で標準化する"""
    n = len(rewards)
    returns_np = np.zeros(n, dtype=np.float32)
    discounted_return = 0.0
    for t in reversed(range(n)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns_np[t] = discounted_return
    returns = tf.constant(returns_np, dtype=tf.float32)
    if standardize:
        mean = tf.reduce_mean(returns)
        std = tf.math.reduce_std(returns)
        returns = (returns - mean) / (std + 1e-8)
    return returns

def optimize_policy(
    episode_states: List[tf.Tensor],
    episode_actions: List[int],
    episode_action_masks: List[tf.Tensor],
    returns: tf.Tensor,
    policy_net: keras.Model,
    optimizer: keras.optimizers.Optimizer
) -> float:
    """勾配テープを使用して方策ネットワークを最適化する"""
    states_tensor = tf.stack(episode_states)
    actions_tensor = tf.constant(episode_actions, dtype=tf.int32)
    masks_tensor = tf.stack(episode_action_masks)

    with tf.GradientTape() as tape:
        # GradientTapeのコンテキスト内で、モデルからlogitsを取得
        all_logits = policy_net(states_tensor, mask=masks_tensor)
        # Categorical分布にlogitsを渡す
        dist = tfp.distributions.Categorical(logits=all_logits)
        log_probs = dist.log_prob(actions_tensor)
        # 損失を計算
        loss = -tf.reduce_sum(returns * log_probs)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    return loss.numpy()

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

# 迷路の構造情報
maze_policy, maze_width, maze_height = create_3x3_maze_policy()
exit_position = 8

# gym.makeで環境を生成
env = gym.make(
    'Maze-v0',
    max_episode_steps=MAX_STEPS_PER_EPISODE_REINFORCE,
    policy=maze_policy, width=maze_width, height=maze_height, exit_pos=exit_position,
)

n_states = env.observation_space.n
n_actions = env.action_space.n
n_observations = n_states

print(f"Gymnasium Maze Environment (One-Hot Input, TimeLimit Wrapper Active):")
print(f"Total States (NN input dim): {n_observations}, Total Actions: {n_actions}")

# Kerasのモデルとオプティマイザを初期化
policy_net_reinforce = PolicyNetwork(n_observations, n_actions)
optimizer_reinforce = keras.optimizers.Adam(learning_rate=LR_REINFORCE)

episode_rewards_reinforce, episode_lengths_reinforce, episode_losses_reinforce = [], [], []

print("\nStarting REINFORCE Training...")

for i_episode in range(NUM_EPISODES_REINFORCE):
    obs, info = env.reset()
    # 状態、行動、マスク、報酬を保存するリスト
    episode_states, episode_actions, episode_action_masks, episode_rewards = [], [], [], []
    terminated, truncated = False, False

    while not (terminated or truncated):
        state_tensor = tf.one_hot(tf.constant(obs), depth=n_states, dtype=tf.float32)
        action_mask = valid_actions_to_mask(info['valid_actions'], n_actions)

        action, _ = select_action_reinforce(state_tensor, policy_net_reinforce, action_mask)

        # エピソードのデータを保存
        episode_states.append(state_tensor)
        episode_actions.append(action)
        episode_action_masks.append(action_mask)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated and not truncated:
            reward = GOAL_REWARD

        episode_rewards.append(reward)
        obs = next_obs

    # 割引収益を計算
    returns = calculate_discounted_returns(episode_rewards, GAMMA_REINFORCE, STANDARDIZE_RETURNS)
    # 方策を最適化
    loss = optimize_policy(
        episode_states,
        episode_actions,
        episode_action_masks,
        returns,
        policy_net_reinforce,
        optimizer_reinforce
    )

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
    n_actions: int
) -> None:
    action_symbols = {0: '↑', 1: '↓', 2: '→', 3: '←'}
    fig, ax = plt.subplots(figsize=(width * 1.2, height * 1.2))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.invert_yaxis()

    for s in range(n_states):
        r, c = divmod(s, width)
        if not policy_map[s, 0]: ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color='k', lw=2)
        if not policy_map[s, 1]: ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color='k', lw=2)
        if not policy_map[s, 2]: ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color='k', lw=2)
        if not policy_map[s, 3]: ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color='k', lw=2)

        if s == goal_pos:
            ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=22, weight='bold')
        elif s == 0:
            ax.text(c, r, 'S', ha='center', va='center', color='red', fontsize=18, weight='bold')
        else:
            state_tensor = tf.one_hot(tf.constant(s), depth=n_states, dtype=tf.float32)
            valid_actions = [i for i, v in enumerate(policy_map[s]) if v]
            if not valid_actions: continue
            action_mask = valid_actions_to_mask(valid_actions, n_actions)

            # モデルからlogitsを取得し、softmaxを適用して確率を計算
            logits = policy_net(tf.expand_dims(state_tensor, 0), mask=action_mask)
            action_probs = tf.nn.softmax(logits)
            best_action = tf.argmax(action_probs, axis=1).numpy()[0]

            ax.text(c, r, action_symbols[best_action], ha='center', va='center', color='blue', fontsize=20)

    ax.set_title("REINFORCE Learned Policy (Grid Corrected)", fontsize=16)
    plt.show()


print("\nPlotting Learned Policy:")
plot_policy_on_maze(
    policy_net=policy_net_reinforce,
    policy_map=maze_policy,
    width=maze_width,
    height=maze_height,
    goal_pos=exit_position,
    n_states=n_states,
    n_actions=n_actions
)

env.close()