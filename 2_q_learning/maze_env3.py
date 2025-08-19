# ==============================================================================
# 0. ライブラリのインポート
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Optional, Any
import time

# Gymnasiumのインポート
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register # ★変更点: registerをインポート
from gymnasium.wrappers import TimeLimit

# 乱数シードの固定
#seed = 42
#random.seed(seed)
#np.random.seed(seed)

# ==============================================================================
# 1. Gymnasium互換の迷路環境 (MazeEnv) の定義
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
        self.maze_map = maze_map  # 壁の情報 (各セルから各方向へ進めるか)
        
        self._start_pos = np.array([0, 0], dtype=np.int32)
        self._goal_pos = np.array([self.height - 1, self.width - 1], dtype=np.int32)
        self._agent_location = np.copy(self._start_pos)

        # === 観測空間と行動空間の定義 ===
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
        # (レンダリング関連の初期化は省略)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """現在の観測（座標とアクションマスク）を生成する"""
        y, x = self._agent_location
        # 現在地から移動可能な方向がTrueになっているマスクを取得
        action_mask = self.maze_map[y, x].astype(np.int8)
        return {
            "agent_position": self._agent_location,
            "action_mask": action_mask
        }

    def _get_info(self) -> Dict[str, Any]:
        """補助情報 (今回は特にないので空)"""
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._agent_location = np.copy(self._start_pos)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # ★変更点: 不正なアクションが渡された場合に例外を発生させる
        y, x = self._agent_location
        if not self.maze_map[y, x, action]:
            raise ValueError(f"Invalid action {action} at position {self._agent_location}. "
                             f"Allowed actions: {np.where(self.maze_map[y, x])[0]}")

        if action == self.UP: self._agent_location[0] -= 1
        elif action == self.DOWN: self._agent_location[0] += 1
        elif action == self.RIGHT: self._agent_location[1] += 1
        elif action == self.LEFT: self._agent_location[1] -= 1
        
        terminated = np.array_equal(self._agent_location, self._goal_pos)
        reward = -1.0  # デフォルトの報酬
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

# ★変更点: 環境をGymnasiumに登録
register(
    id='Maze-v0',
    entry_point=MazeEnv,
)

def create_3x3_maze_map() -> np.ndarray:
    """
    3x3迷路の壁情報（移動可能か）を生成する。
    元のコードの定義を安全に変換して使用する。
    """
    height, width = 3, 3
    
    # 元のコードの行動定義: UP=0, DOWN=1, RIGHT=2, LEFT=3
    original_actions = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
    original_allowed_moves = {
        0:[1,2], 1:[1,2,3], 2:[3], 3:[0,1], 4:[0,2], 5:[1,3], 6:[0,2], 7:[3], 8:[0]
    }
    
    # 新しいコードの行動定義: UP=0, RIGHT=1, DOWN=2, LEFT=3
    # このマッピングに従って変換する
    new_action_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # (height, width, 4) の形状。4は[UP, RIGHT, DOWN, LEFT]への移動可否
    maze_map = np.zeros((height, width, 4), dtype=bool)

    for state_id, actions in original_allowed_moves.items():
        y, x = divmod(state_id, width)
        for old_action_idx in actions:
            # 古い行動名を新しい行動インデックスに変換
            action_name = original_actions[old_action_idx]
            new_action_idx = new_action_map[action_name]
            maze_map[y, x, new_action_idx] = True
            
    return maze_map

def softmax(x):
    # オーバーフロー防止のため max を引くのが定番
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

if __name__ == '__main__':
    # 環境の生成
    maze_map_3x3 = create_3x3_maze_map()

    # ★変更点: gym.make() を使って環境を生成
    #env = gym.make('Maze-v0', maze_map=maze_map_3x3)
    ## TimeLimitラッパーで最大ステップ数を設定
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE_REINFORCE)

    height, width, exit_pos = 3, 3, 8

    env = gym.make(
        'Maze-v0',
        maze_map=maze_map_3x3,
        render_mode='human',
        max_episode_steps=100,  # ★★★ ここで最大ステップ数を指定 ★★★
        # 以下はMazeEnvの__init__に渡される引数
    )
    num_actions = env.action_space.n

    # TimeLimitラッパーが適用されていることを確認 (オプション)
    print(f"環境のラッパー: {env}")
    assert isinstance(env, TimeLimit)

    # 環境をリセット
    observation, info = env.reset()
    print(f"初期状態: {observation['agent_position']}, 有効な行動: {observation['action_mask']}")

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
        prob = softmax(-1e9*(1-observation['action_mask']))
        action = np.random.choice(range(num_actions),p=prob)
        print('action=',action)

        # 1ステップ進める
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

        # 状態を出力
        print(f"ステップ {step_count:3d}: 行動={action} -> 状態={observation['agent_position']},{observation['action_mask']}, 報酬={reward}, term={terminated}, trunc={truncated}")

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
