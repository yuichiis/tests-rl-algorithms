# %%
# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable, Any, Union
import copy
import os
import time

# TensorFlowをインポート
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Gymnasium（強化学習環境）をインポート
try:
    import gymnasium as gym
except ImportError:
    print("Gymnasiumが見つかりません。'pip install gymnasium' または 'pip install gym[classic_control]' でインストールしてください。")
    # gymが必須の場合はここで終了またはエラーを発生させる
    gym = None

# %%
# デバイス設定
# TensorFlowは利用可能な場合に自動的にGPUを使用します
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 現在、メモリ成長はデフォルトで有効になっていることが多い
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"利用可能なGPU: {len(gpus)}台")
    except RuntimeError as e:
        print(e)
else:
    print("GPUが見つかりません。CPUを使用します。")

# 再現性のための乱数シード設定
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %%
# Pendulum環境のインスタンス化
if gym is not None:
    try:
        # 環境の作成
        env = gym.make('Pendulum-v1')
        
        # 環境の再現性のためのシード設定
        env.reset(seed=seed)
        env.action_space.seed(seed)

        # 状態と行動空間の次元を取得
        n_observations_ddpg = env.observation_space.shape[0]
        n_actions_ddpg = env.action_space.shape[0] # DDPGは連続値行動を扱う
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        print(f"Pendulum 環境:")
        print(f"状態空間の次元: {n_observations_ddpg}")
        print(f"行動空間の次元: {n_actions_ddpg}")
        print(f"行動の下限: {action_low}")
        print(f"行動の上限: {action_high}")
        
        # resetのテスト
        obs, info = env.reset()
        print(f"初期観測値: {obs}")
        
    except Exception as e:
        print(f"Gymnasium環境の作成中にエラーが発生しました: {e}")
        # 環境作成失敗時のダミー値
        n_observations_ddpg = 3
        n_actions_ddpg = 1
        action_low = -2.0
        action_high = 2.0
        env = None # envを使用不可にマーク
else:
    print("Gymnasiumが利用できません。Pendulum環境を作成できません。")
    # ダミー値の設定
    n_observations_ddpg = 3
    n_actions_ddpg = 1
    action_low = -2.0
    action_high = 2.0
    env = None


# %%
# Actorネットワークの定義
class ActorNetwork(keras.Model):
    """ DDPGのための決定論的Actorネットワーク """
    def __init__(self, n_actions: int, action_high_bound: float):
        super(ActorNetwork, self).__init__()
        self.action_high_bound = action_high_bound
        # 最終層の重みを小さな値で初期化するためのInitialier
        last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        
        # シンプルなMLPアーキテクチャ
        self.layer1 = layers.Dense(256, activation='relu')
        self.layer2 = layers.Dense(256, activation='relu')
        self.layer3 = layers.Dense(n_actions, activation='tanh',
                                   kernel_initializer=last_init,
                                   bias_initializer=last_init)
    
    def call(self, state: tf.Tensor) -> tf.Tensor:
        """
        状態を決定論的な行動にマッピングする
        引数:
        - state (tf.Tensor): 入力状態テンソル
        戻り値:
        - tf.Tensor: 環境の行動範囲にスケーリングされた決定論的な行動
        """
        x = self.layer1(state)
        x = self.layer2(x)
        # tanhで出力を-1と1の間に制限
        action_tanh = self.layer3(x)
        # 環境の行動範囲にスケーリング
        scaled_action = action_tanh * self.action_high_bound
        return scaled_action

# %%
# Criticネットワークの定義
class CriticNetwork(keras.Model):
    """ DDPGのためのQ値Criticネットワーク """
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # 最初に状態を別々に処理
        self.state_layer1 = layers.Dense(256, activation='relu')
        # 状態の特徴量と行動を2番目の層で結合
        self.combined_layer2 = layers.Dense(256, activation='relu')
        self.output_layer3 = layers.Dense(1) # Q値の出力層

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        状態と行動をQ値にマッピングする
        引数:
        - inputs (Tuple[tf.Tensor, tf.Tensor]): (状態テンソル, 行動テンソル)
        戻り値:
        - tf.Tensor: 推定されたQ(s, a)値
        """
        state, action = inputs
        state_features = self.state_layer1(state)
        # 状態の特徴量と行動を結合
        combined = tf.concat([state_features, action], axis=1)
        x = self.combined_layer2(combined)
        q_value = self.output_layer3(x)
        return q_value

# %%
# リプレイメモリの定義
# 遷移を保存するための構造を定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    """ 遷移を保存し、バッチのサンプリングを可能にする """
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# %%
# 探索ノイズの定義
class GaussianNoise:
    def __init__(self, action_dimension: int, mean: float = 0.0, std_dev: float = 0.1):
        self.action_dim = action_dimension
        self.mean = mean
        self.std_dev = std_dev

    def get_noise(self) -> np.ndarray:
        return np.random.normal(self.mean, self.std_dev, self.action_dim)

    def reset(self) -> None:
        pass

# %%
# ######################################################################
# # ここを修正しました
# ######################################################################
# Soft Update関数 (修正版)
def soft_update(target_net: keras.Model, main_net: keras.Model, tau: float) -> None:
    """
    ターゲットネットワークのパラメータをソフトアップデートする。
    このバージョンはtf.Variableを直接操作するため、@tf.function内で動作する。
    θ_target = τ*θ_local + (1 - τ)*θ_target

    引数:
    - target_net (keras.Model): 更新されるターゲットネットワーク
    - main_net (keras.Model): パラメータを提供するメインネットワーク
    - tau (float): ソフトアップデート係数 (τ)
    """
    # .get_weights()/.set_weights() を使わず、変数を直接操作する
    for main_var, target_var in zip(main_net.trainable_variables, target_net.trainable_variables):
        target_var.assign(tau * main_var + (1.0 - tau) * target_var)
# ######################################################################

# %%
# DDPGの更新ステップ
@tf.function
def update_ddpg(
    state_batch: tf.Tensor,
    action_batch: tf.Tensor,
    reward_batch: tf.Tensor,
    next_state_batch: tf.Tensor,
    done_batch: tf.Tensor,
    actor: ActorNetwork,
    critic: CriticNetwork,
    target_actor: ActorNetwork,
    target_critic: CriticNetwork,
    actor_optimizer: keras.optimizers.Optimizer,
    critic_optimizer: keras.optimizers.Optimizer,
    gamma: float,
    tau: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    DDPGの更新ステップ（ActorとCritic）を1回実行する
    """
    # --- Criticの更新 ---
    with tf.GradientTape() as tape:
        next_actions = target_actor(next_state_batch, training=True)
        target_q_values = target_critic([next_state_batch, next_actions], training=True)
        y = reward_batch + gamma * (1.0 - done_batch) * target_q_values
        current_q_values = critic([state_batch, action_batch], training=True)
        critic_loss = tf.reduce_mean(tf.square(y - current_q_values))

    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # --- Actorの更新 ---
    with tf.GradientTape() as tape:
        actor_actions = actor(state_batch, training=True)
        q_values_for_actor = critic([state_batch, actor_actions], training=True)
        actor_loss = -tf.reduce_mean(q_values_for_actor)
    
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    # --- ターゲットネットワークの更新 ---
    # 修正されたsoft_update関数を呼び出す
    soft_update(target_critic, critic, tau)
    soft_update(target_actor, actor, tau)

    return critic_loss, actor_loss

# %%
# DDPGアルゴリズムの実行
# ハイパーパラメータ
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA_DDPG = 0.99
TAU = 1e-3
ACTOR_LR_DDPG = 1e-4
CRITIC_LR_DDPG = 1e-3
WEIGHT_DECAY = 1e-2 
NOISE_STD_DEV = 0.2
NOISE_DECAY = 0.999
MIN_NOISE_STD_DEV = 0.01
NUM_EPISODES_DDPG = 100
MAX_STEPS_PER_EPISODE_DDPG = 500
UPDATE_EVERY = 1
NUM_UPDATES = 1

# 初期化
if env is None:
    raise RuntimeError("Gymnasium環境 'Pendulum-v1' が作成できませんでした。")

actor_ddpg = ActorNetwork(n_actions_ddpg, action_high)
critic_ddpg = CriticNetwork()

target_actor_ddpg = ActorNetwork(n_actions_ddpg, action_high)
target_critic_ddpg = CriticNetwork()

# ネットワークを一度ビルドしてから重みをコピーする
dummy_state = tf.random.uniform((1, n_observations_ddpg))
dummy_action = tf.random.uniform((1, n_actions_ddpg))
actor_ddpg(dummy_state)
critic_ddpg([dummy_state, dummy_action])
target_actor_ddpg(dummy_state)
target_critic_ddpg([dummy_state, dummy_action])

target_actor_ddpg.set_weights(actor_ddpg.get_weights())
target_critic_ddpg.set_weights(critic_ddpg.get_weights())

actor_optimizer_ddpg = keras.optimizers.Adam(learning_rate=ACTOR_LR_DDPG)
critic_optimizer_ddpg = keras.optimizers.AdamW(learning_rate=CRITIC_LR_DDPG, weight_decay=WEIGHT_DECAY)

memory_ddpg = ReplayMemory(BUFFER_SIZE)
noise = GaussianNoise(n_actions_ddpg, std_dev=NOISE_STD_DEV)
current_noise_std_dev = NOISE_STD_DEV

ddpg_episode_rewards = []
ddpg_episode_actor_losses = []
ddpg_episode_critic_losses = []

# %%
# 学習ループ
print("Pendulum-v1でのDDPG学習を開始します...")

total_steps = 0
for i_episode in range(1, NUM_EPISODES_DDPG + 1):
    state_np, info = env.reset()
    noise.reset()
    noise.std_dev = current_noise_std_dev
    
    episode_reward = 0
    actor_losses = []
    critic_losses = []

    for t in range(MAX_STEPS_PER_EPISODE_DDPG):
        state_tf = tf.expand_dims(tf.convert_to_tensor(state_np, dtype=tf.float32), 0)
        action_deterministic = actor_ddpg(state_tf, training=False)
        action_deterministic_np = action_deterministic.numpy()[0]
        
        action_noise = noise.get_noise()
        action_noisy = action_deterministic_np + action_noise
        action_clipped = np.clip(action_noisy, action_low, action_high)

        next_state_np, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated
        
        memory_ddpg.push(state_np, action_clipped, reward, next_state_np, done)

        state_np = next_state_np
        episode_reward += reward
        total_steps += 1

        if len(memory_ddpg) > BATCH_SIZE and total_steps % UPDATE_EVERY == 0:
            for _ in range(NUM_UPDATES):
                transitions = memory_ddpg.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                state_batch_np = np.vstack(batch.state)
                action_batch_np = np.vstack(batch.action)
                reward_batch_np = np.array(batch.reward, dtype=np.float32).reshape(-1, 1)
                next_state_batch_np = np.vstack(batch.next_state)
                done_batch_np = np.array(batch.done, dtype=np.float32).reshape(-1, 1)

                state_batch = tf.convert_to_tensor(state_batch_np, dtype=tf.float32)
                action_batch = tf.convert_to_tensor(action_batch_np, dtype=tf.float32)
                reward_batch = tf.convert_to_tensor(reward_batch_np, dtype=tf.float32)
                next_state_batch = tf.convert_to_tensor(next_state_batch_np, dtype=tf.float32)
                done_batch = tf.convert_to_tensor(done_batch_np, dtype=tf.float32)

                c_loss, a_loss = update_ddpg(
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                    actor_ddpg, critic_ddpg,
                    target_actor_ddpg, target_critic_ddpg,
                    actor_optimizer_ddpg, critic_optimizer_ddpg,
                    GAMMA_DDPG, TAU
                )
                critic_losses.append(c_loss.numpy())
                actor_losses.append(a_loss.numpy())

        if done:
            break
            
    ddpg_episode_rewards.append(episode_reward)
    ddpg_episode_actor_losses.append(np.mean(actor_losses) if actor_losses else 0)
    ddpg_episode_critic_losses.append(np.mean(critic_losses) if critic_losses else 0)
    
    current_noise_std_dev = max(MIN_NOISE_STD_DEV, current_noise_std_dev * NOISE_DECAY)
    
    if i_episode % 10 == 0:
        avg_reward = np.mean(ddpg_episode_rewards[-10:])
        avg_actor_loss = np.mean(ddpg_episode_actor_losses[-10:])
        avg_critic_loss = np.mean(ddpg_episode_critic_losses[-10:])
        print(f"エピソード {i_episode}/{NUM_EPISODES_DDPG} | 平均報酬: {avg_reward:.2f} | Actor損失: {avg_actor_loss:.4f} | Critic損失: {avg_critic_loss:.4f} | ノイズ標準偏差: {current_noise_std_dev:.3f}")

print("Pendulum-v1での学習が完了しました (DDPG).")

# %%
# 学習プロセスの可視化
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.plot(ddpg_episode_rewards)
plt.title('DDPG Pendulum: Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
if len(ddpg_episode_rewards) >= 10:
    rewards_ma_ddpg = np.convolve(ddpg_episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(rewards_ma_ddpg)) + 9, rewards_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

plt.subplot(1, 3, 2)
plt.plot(ddpg_episode_critic_losses)
plt.title('DDPG Pendulum: Avg Critic Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('MSE Loss')
plt.grid(True)
if len(ddpg_episode_critic_losses) >= 10:
    closs_ma_ddpg = np.convolve(ddpg_episode_critic_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(closs_ma_ddpg)) + 9, closs_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

plt.subplot(1, 3, 3)
plt.plot(ddpg_episode_actor_losses)
plt.title('DDPG Pendulum: Avg Actor Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Avg -Q Value')
plt.grid(True)
if len(ddpg_episode_actor_losses) >= 10:
    aloss_ma_ddpg = np.convolve(ddpg_episode_actor_losses, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(len(aloss_ma_ddpg)) + 9, aloss_ma_ddpg, label='10-episode MA', color='orange')
    plt.legend()

plt.tight_layout()
plt.show()

# %%
# 学習済み方策の分析（テスト）
def test_ddpg_agent(actor_net: keras.Model, 
                    env_instance: gym.Env, 
                    num_episodes: int = 5, 
                    render: bool = False,
                    seed_offset: int = 1000) -> None:
    if env_instance is None:
        print("テストに利用できる環境がありません。")
        return
        
    print(f"\n--- DDPGエージェントのテスト ({num_episodes} エピソード) ---")
    all_rewards = []
    for i in range(num_episodes):
        state_np, info = env_instance.reset(seed=seed + seed_offset + i)
        episode_reward = 0
        done = False
        t = 0
        while not done:
            if render:
                try:
                    env_instance.render()
                    time.sleep(0.01)
                except Exception as e:
                    print(f"描画に失敗しました: {e}。描画を無効にします。")
                    render = False
            
            state_tf = tf.expand_dims(tf.convert_to_tensor(state_np, dtype=tf.float32), 0)
            action = actor_net(state_tf, training=False).numpy()[0]
            action_clipped = np.clip(action, env_instance.action_space.low, env_instance.action_space.high)
            
            next_state_np, reward, terminated, truncated, _ = env_instance.step(action_clipped)
            done = terminated or truncated
            state_np = next_state_np
            episode_reward += reward
            t += 1
        
        print(f"テストエピソード {i+1}: 報酬 = {episode_reward:.2f}, 長さ = {t}")
        all_rewards.append(episode_reward)
        if render:
            env_instance.close()

    print(f"--- テスト完了。平均報酬: {np.mean(all_rewards):.2f} ---")

env = gym.make('Pendulum-v1', render_mode='human')
test_ddpg_agent(actor_ddpg, env, num_episodes=3, render=False)