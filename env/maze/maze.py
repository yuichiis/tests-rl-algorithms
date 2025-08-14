import numpy as np
from maze_env import MazeEnv
import time

def create_3x3_maze_policy():
    """
    簡単な3x3の迷路のポリシー（壁情報）を作成します。
    +---+---+---+
    | S   1   2 | (S: Start at 0)
    +   +   +---+
    | 3 | 4   5 |
    +   +---+   +
    | 6   7 | G | (G: Goal at 8)
    +-------+---+
    """
    width, height = 3, 3
    num_states = width * height
    # policy[state, action] = Trueなら移動可能
    # actions: 0:UP, 1:DOWN, 2:RIGHT, 3:LEFT
    policy = np.zeros((num_states, 4), dtype=bool)

    # 各セルからの移動可能性を定義
    # (state, allowed_actions)
    allowed_moves = {
        0: [1, 2], 1: [1, 2, 3], 2: [3],
        3: [0, 1], 4: [0, 1],    5: [1, 3],
        6: [0, 2], 7: [2],       8: [0]
    }
    
    for state, actions in allowed_moves.items():
        for action in actions:
            policy[state, action] = True
            
    return policy, width, height

if __name__ == '__main__':
    # 迷路を作成
    policy, width, height = create_3x3_maze_policy()
    exit_pos = 8

    # 環境を初期化
    # 'human'モードでレンダリングを有効化
    env = MazeEnv(policy, width, height, exit_pos, render_mode='human')

    # 環境をリセット
    observation, info = env.reset()
    print(f"初期状態: {observation}, 有効な行動: {info['valid_actions']}")

    terminated = False
    total_reward = 0
    
    # 20ステップ、ランダムに行動
    for i in range(20):
        if terminated:
            print(f"ゴールに到達しました！ Total Reward: {total_reward}")
            break

        # 有効な行動の中からランダムに選択
        action = np.random.choice(info['valid_actions'])
        
        # 1ステップ進める
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        print(f"ステップ {i+1}: 行動={action} -> 状態={observation}, 報酬={reward}, 終了={terminated}")
        print(f"  有効な行動: {info.get('valid_actions')}")

        # 描画が追いつくように少し待つ
        time.sleep(0.2)
        
    if not terminated:
        print("20ステップ以内にゴールできませんでした。")

    # 環境を閉じる
    env.close()