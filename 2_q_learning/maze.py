import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import random
import time

# ------------- ユーティリティ -----------------
def state_to_id(pos, width):
    """座標(y,x)を状態IDに変換"""
    y, x = pos
    return y * width + x

def epsilon_greedy(Q, state_id, action_mask, epsilon):
    """ε-greedyで行動を選択（無効アクションは除外）"""
    valid_actions = np.where(action_mask == 1)[0]
    if np.random.rand() < epsilon:
        return np.random.choice(valid_actions)
    else:
        q_values = Q[state_id].copy()
        # 無効アクションは -1e9 にして argmax から除外
        q_values[action_mask == 0] = -1e9
        return np.argmax(q_values)

# ------------- Q-Learning 本体 -----------------
def q_learning(env, height, width, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.2):
    #height, width = env.height, env.width
    num_states = height * width
    num_actions = env.action_space.n
    
    # Qテーブル初期化
    Q = np.zeros((num_states, num_actions))
    
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state_id = state_to_id(obs["agent_position"], width)
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = epsilon_greedy(Q, state_id, obs["action_mask"], epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_id = state_to_id(next_obs["agent_position"], width)
            
            # 次状態の価値（無効アクションは除外）
            next_q = Q[next_state_id].copy()
            next_q[next_obs["action_mask"] == 0] = -1e9
            max_next_q = np.max(next_q)
            
            # Q値更新
            Q[state_id, action] += alpha * (reward + gamma * max_next_q - Q[state_id, action])
            
            state_id = next_state_id
            obs = next_obs
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        # 学習の進行を少し表示
        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, total_reward={total_reward}")
    
    return Q, rewards_per_episode

# ------------- 実行 -----------------
if __name__ == '__main__':
    from maze_env3 import create_3x3_maze_map, MazeEnv
    
    maze_map_3x3 = create_3x3_maze_map()
    env = gym.make('Maze-v0', maze_map=maze_map_3x3, max_episode_steps=100)
    height, width = maze_map_3x3.shape[:2]
    num_actions = env.action_space.n
    
    Q, rewards = q_learning(env, height, width, num_episodes=500)
    
    print("学習終了！")
    print("学習済みQテーブル:")
    print(Q.reshape(height, width, num_actions))

