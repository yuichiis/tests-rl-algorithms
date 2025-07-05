# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque
from typing import List, Tuple, Dict

# TensorFlowをインポート
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses

# TensorFlowが利用可能なGPUを検出して使用するように設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Using device: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Using device: CPU")

# 再現性のためのランダムシードを設定
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%matplotlib inline

# --- 修正点: 環境がtf.Tensorを返すように変更 ---
class GridEnvironment:
    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        self.rows, self.cols = rows, cols
        self.start_state, self.goal_state = (0, 0), (rows - 1, cols - 1)
        self.state = self.start_state
        self.state_dim, self.action_dim = 2, 4
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self) -> tf.Tensor:
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> tf.Tensor:
        norm_row = state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0
        norm_col = state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        return tf.convert_to_tensor([norm_row, norm_col], dtype=tf.float32)

    def step(self, action: int) -> Tuple[tf.Tensor, float, bool]:
        if self.state == self.goal_state: return self._get_state_tensor(self.state), 0.0, True
        dr, dc = self.action_map[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        reward = -0.1
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc, reward = r, c, -1.0
        self.state = (nr, nc)
        done = (self.state == self.goal_state)
        if done: reward = 10.0
        return self._get_state_tensor(self.state), reward, done

    def get_action_space_size(self) -> int: return self.action_dim
    def get_state_dimension(self) -> int: return self.state_dim

# A2Cエージェントクラス
class A2CAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, gae_lambda, entropy_coeff):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff

        self.actor = self._build_network(state_dim, action_dim)
        self.critic = self._build_network(state_dim, 1)

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

    def _build_network(self, input_dim, output_dim):
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(output_dim)(x)
        return Model(inputs, outputs)

    # --- 修正点: アクション選択も@tf.functionで高速化 ---
    @tf.function
    def get_action_and_value_tf(self, state):
        state_batch = tf.expand_dims(state, 0)
        logits = self.actor(state_batch)
        value = self.critic(state_batch)
        action = tf.random.categorical(logits, 1)[0, 0]
        return action, tf.squeeze(value)

    # numpyに変換するためのラッパー
    def get_action_and_value(self, state_tensor):
        action, value = self.get_action_and_value_tf(state_tensor)
        return action.numpy(), value.numpy()
        
    @tf.function
    def get_value_tf(self, state):
        return tf.squeeze(self.critic(tf.expand_dims(state, 0)))

    # train_stepは変更なしでも高速化の恩恵を受ける
    @tf.function
    def train_step(self, states, actions, advantages, returns):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            logits = self.actor(states, training=True)
            values_pred = tf.squeeze(self.critic(states, training=True), axis=1)

            neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            policy_loss = tf.reduce_mean(neg_log_probs * advantages)

            probs = tf.nn.softmax(logits)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=1))

            actor_total_loss = policy_loss - self.entropy_coeff * entropy_loss
            critic_loss = tf.reduce_mean(losses.mean_squared_error(returns, values_pred))

        actor_grads = actor_tape.gradient(actor_total_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return policy_loss, critic_loss, entropy_loss

# --- メインのトレーニングループ ---
# ハイパーパラメータ
GAMMA, GAE_LAMBDA = 0.99, 0.95
ACTOR_LR, CRITIC_LR = 3e-4, 1e-3
ENTROPY_COEFF = 0.01
NUM_ITERATIONS, STEPS_PER_ITERATION, MAX_STEPS_PER_EPISODE = 400, 256, 200

# 環境とエージェントの初期化
env = GridEnvironment(rows=10, cols=10)
agent = A2CAgent(env.get_state_dimension(), env.get_action_space_size(), ACTOR_LR, CRITIC_LR, GAMMA, GAE_LAMBDA, ENTROPY_COEFF)

# ログ用リスト
iteration_rewards, iteration_avg_ep_lens, iteration_policy_losses, iteration_value_losses, iteration_entropies = [], [], [], [], []

print("Starting A2C Training on Custom Grid World...")
start_time = time.time()
state = env.reset() # stateはtf.Tensor

for iteration in range(NUM_ITERATIONS):
    batch_states, batch_actions, batch_rewards, batch_values, batch_dones = [], [], [], [], []
    episode_rewards_in_iter, episode_lengths_in_iter = [], []
    current_episode_reward, current_episode_length = 0.0, 0

    for _ in range(STEPS_PER_ITERATION):
        action, value = agent.get_action_and_value(state) # stateはtf.Tensor
        next_state, reward, done = env.step(action) # next_stateもtf.Tensor
        
        batch_states.append(state)
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_values.append(value)
        batch_dones.append(done)

        state = next_state
        current_episode_reward += reward
        current_episode_length += 1

        if done or current_episode_length >= MAX_STEPS_PER_EPISODE:
            episode_rewards_in_iter.append(current_episode_reward)
            episode_lengths_in_iter.append(current_episode_length)
            state = env.reset()
            current_episode_reward, current_episode_length = 0.0, 0
    
    last_value = agent.get_value_tf(state).numpy() if not done else 0.0
    
    rewards_tensor = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
    values_tensor = tf.convert_to_tensor(batch_values, dtype=tf.float32)
    next_values_tensor = tf.convert_to_tensor(batch_values[1:] + [last_value], dtype=tf.float32)
    dones_tensor = tf.convert_to_tensor(batch_dones, dtype=tf.float32)

    advantages_reversed = []
    last_advantage = 0.0
    for t in reversed(range(STEPS_PER_ITERATION)):
        mask = 1.0 - dones_tensor[t]
        delta = rewards_tensor[t] + GAMMA * next_values_tensor[t] * mask - values_tensor[t]
        advantage = delta + GAMMA * GAE_LAMBDA * last_advantage * mask
        advantages_reversed.append(advantage)
        last_advantage = advantage
    advantages = tf.stack(list(reversed(advantages_reversed)))
    returns = advantages + values_tensor
    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
    
    states_tensor = tf.stack(batch_states)
    actions_tensor = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
    
    p_loss, v_loss, entropy = agent.train_step(states_tensor, actions_tensor, advantages, returns)

    avg_reward = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
    avg_len = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan
    iteration_rewards.append(avg_reward)
    iteration_avg_ep_lens.append(avg_len)
    iteration_policy_losses.append(p_loss.numpy())
    iteration_value_losses.append(v_loss.numpy())
    iteration_entropies.append(entropy.numpy())

    if (iteration + 1) % 50 == 0:
        print(f"Iter {iteration+1}/{NUM_ITERATIONS} | Avg Reward: {avg_reward:.2f} | Avg Len: {avg_len:.1f} | P_Loss: {p_loss.numpy():.4f} | V_Loss: {v_loss.numpy():.4f} | Entropy: {entropy.numpy():.4f}")

end_time = time.time()
print(f"Custom Grid World Training Finished (A2C). Total time: {end_time - start_time:.2f} seconds.")

# プロット
plt.figure(figsize=(20, 8))
# (省略... 元のコードと同じプロットコード)
plt.subplot(2, 3, 1)
valid_rewards = [r for r in iteration_rewards if not np.isnan(r)]
valid_indices = [i for i, r in enumerate(iteration_rewards) if not np.isnan(r)]
plt.plot(valid_indices, valid_rewards)
plt.title('A2C Custom Grid: Avg Ep Reward / Iteration'); plt.xlabel('Iteration'); plt.ylabel('Avg Reward'); plt.grid(True)
if len(valid_rewards) >= 20: plt.plot(valid_indices[19:], np.convolve(valid_rewards, np.ones(20)/20, mode='valid'), label='20-iter MA', color='orange'); plt.legend()
plt.subplot(2, 3, 2)
valid_lens = [l for l in iteration_avg_ep_lens if not np.isnan(l)]
valid_indices_len = [i for i, l in enumerate(iteration_avg_ep_lens) if not np.isnan(l)]
plt.plot(valid_indices_len, valid_lens)
plt.title('A2C Custom Grid: Avg Ep Length / Iteration'); plt.xlabel('Iteration'); plt.ylabel('Avg Steps'); plt.grid(True)
if len(valid_lens) >= 20: plt.plot(valid_indices_len[19:], np.convolve(valid_lens, np.ones(20)/20, mode='valid'), label='20-iter MA', color='orange'); plt.legend()
plt.subplot(2, 3, 3)
plt.plot(iteration_value_losses); plt.title('A2C Custom Grid: Value Loss / Iteration'); plt.xlabel('Iteration'); plt.ylabel('MSE Loss'); plt.grid(True)
if len(iteration_value_losses) >= 20: plt.plot(np.arange(len(iteration_value_losses)-19)+19, np.convolve(iteration_value_losses, np.ones(20)/20, mode='valid'), label='20-iter MA', color='orange'); plt.legend()
plt.subplot(2, 3, 4)
plt.plot(iteration_policy_losses); plt.title('A2C Custom Grid: Policy Loss / Iteration'); plt.xlabel('Iteration'); plt.ylabel('Policy Objective'); plt.grid(True)
if len(iteration_policy_losses) >= 20: plt.plot(np.arange(len(iteration_policy_losses)-19)+19, np.convolve(iteration_policy_losses, np.ones(20)/20, mode='valid'), label='20-iter MA', color='orange'); plt.legend()
plt.subplot(2, 3, 5)
plt.plot(iteration_entropies); plt.title('A2C Custom Grid: Policy Entropy / Iteration'); plt.xlabel('Iteration'); plt.ylabel('Entropy'); plt.grid(True)
if len(iteration_entropies) >= 20: plt.plot(np.arange(len(iteration_entropies)-19)+19, np.convolve(iteration_entropies, np.ones(20)/20, mode='valid'), label='20-iter MA', color='orange'); plt.legend()
plt.tight_layout(); plt.show()
