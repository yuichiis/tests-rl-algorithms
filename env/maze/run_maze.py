import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import time

# --- この部分は前回の maze_env.py の内容をインポート ---
# このファイルと同じディレクトリに maze_env.py があることを想定
from env.maze.maze_env import MazeEnv 

# ----------------------------------------------------

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
        0: [1, 2],
        1: [1, 2, 3],
        2: [3],
        3: [0, 1],
        4: [0, 2],
        5: [1, 3],
        6: [0, 2],
        7: [3],
        8: [0]
    }
    
    for state, actions in allowed_moves.items():
        for action in actions:
            policy[state, action] = True
            
    return policy, width, height


# --- Gymnasiumへの環境登録 ---
# gym.make()で環境を呼び出せるように、IDを付けて登録します。
gym.register(
     id='Maze-v0',                                  # 環境のID
     entry_point='maze_env:MazeEnv',                # 環境クラスの場所 'ファイル名:クラス名'
)

if __name__ == '__main__':
    # 迷路の定義を作成
    policy, width, height = create_3x3_maze_policy()
    exit_pos = 8

    # --- 環境の生成 ---
    # gym.make() を使って環境を生成し、max_episode_steps を指定します。
    # MazeEnvの__init__引数はkwargsとして渡します。
    print("最大ステップ数100で環境を生成します...")
    env = gym.make(
        'Maze-v0',
        render_mode='human',
        max_episode_steps=100,  # ★★★ ここで最大ステップ数を指定 ★★★
        # 以下はMazeEnvの__init__に渡される引数
        policy=policy,
        width=width,
        height=height,
        exit_pos=exit_pos,
    )

    # TimeLimitラッパーが適用されていることを確認 (オプション)
    print(f"環境のラッパー: {env}")
    assert isinstance(env, TimeLimit)

    # 環境をリセット
    observation, info = env.reset()
    print(f"初期状態: {observation}, 有効な行動: {info['valid_actions']}")

    terminated = False
    truncated = False # 打ち切りフラグ
    total_reward = 0
    step_count = 0

    # 110回ループして、100回で打ち切られることを確認する
    max_steps = 110
    print(f"{max_steps}ステップのシミュレーションを開始...")

    while not terminated and not truncated:
        step_count += 1

        # 有効な行動の中からランダムに選択
        action = np.random.choice(info['valid_actions'])
        
        # 1ステップ進める
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

        # 状態を出力
        print(f"ステップ {step_count:3d}: 行動={action} -> 状態={observation:2d}, 報酬={reward}, term={terminated}, trunc={truncated}")

        #time.sleep(0.05) # 早く実行するためにコメントアウト

    # --- ループ終了後の結果表示 ---
    print("\n--- シミュレーション終了 ---")
    print(f"総ステップ数: {step_count}")
    print(f"総報酬: {total_reward}")

    if terminated:
        print("結果: ゴールに到達しました！")
    elif truncated:
        print("結果: 最大ステップ数(100)に到達したため、エピソードが打ち切られました。")
    else:
        # このケースは通常発生しない
        print("結果: 不明な理由で終了しました。")
    

    time.sleep(10.0) # 早く実行するためにコメントアウト

    # 環境を閉じる
    env.close()