import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any, List

class MazeEnv(gym.Env):
    """
    PHPで実装された迷路環境のGymnasium互換Python版。

    この環境は、グリッド状の迷路を表します。エージェントは各セル（状態）から
    上下左右の4方向に行動できますが、壁によって移動が制限されます。

    - observation_space: 離散値。各セルが一意の整数で表現されます (0, 1, ..., width*height-1)。
    - action_space: 離散値 (0:上, 1:下, 2:右, 3:左)。
    - reward: ゴールに到達するまで、各ステップで-1.0。
    - termination: エージェントがゴールセルに到達したとき。
    - truncation: この環境では発生しません（TimeLimitラッパーで対応）。
    - info: 各ステップで、現在の状態で可能な行動のリスト 'valid_actions' を含みます。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # 行動の定義
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    def __init__(
        self,
        policy: np.ndarray,
        width: int,
        height: int,
        exit_pos: int,
        render_mode: Optional[str] = None,
        throw_invalid_action: bool = True
    ):
        """
        環境を初期化します。

        Args:
            policy (np.ndarray): 迷路の壁情報を表すブール配列。
                                形状は (状態数, 行動数) で、policy[state, action]が
                                Trueならその行動は可能、Falseなら壁で不可能。
            width (int): 迷路の幅。
            height (int): 迷路の高さ。
            exit_pos (int): ゴールセルの位置（状態）。
            render_mode (str, optional): レンダリングモード ('human' or 'rgb_array')。
            throw_invalid_action (bool): 不可能な行動が選択されたときに例外を投げるか。
                                         Falseの場合、報酬-1.0でエピソードを終了します。
        """
        super().__init__()

        if not isinstance(policy, np.ndarray) or policy.ndim != 2 or policy.dtype != bool:
            raise ValueError("policyはブール型の2次元numpy配列である必要があります。")

        self.width = width
        self.height = height
        self.policy = policy
        self.exit_pos = exit_pos
        self.throw_invalid_action = throw_invalid_action

        self.num_states, self.num_actions = self.policy.shape
        if self.num_states != self.width * self.height:
            raise ValueError("policyの行数が width * height と一致しません。")
        if self.num_actions != 4:
            raise ValueError("policyの列数（行動数）は4である必要があります。")

        # Gymnasiumの必須要素を定義
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        # 状態
        self.state: Optional[int] = None

        # レンダリング関連
        self.render_mode = render_mode
        self.window_size = 512  # レンダリングウィンドウのサイズ
        self.window = None
        self.clock = None

    def _get_obs(self) -> int:
        """現在の状態を観測として返す"""
        return self.state

    def _get_info(self) -> Dict[str, Any]:
        """現在の状態で有効な行動を情報として返す"""
        valid_actions_flags = self.policy[self.state]
        valid_actions = [i for i, v in enumerate(valid_actions_flags) if v]
        return {"valid_actions": valid_actions}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        環境を初期状態にリセットします。

        Args:
            seed (int, optional): 乱数生成器のシード。
            options (dict, optional): 追加のオプション（この環境では未使用）。

        Returns:
            tuple: (初期観測, 情報辞書)
        """
        super().reset(seed=seed)
        # エージェントの開始位置は常に0
        self.state = 0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _next_state(self, state: int, action: int) -> int:
        """指定された行動に基づいて次の状態を計算します"""
        if action == self.UP:
            return state - self.width
        if action == self.DOWN:
            return state + self.width
        if action == self.RIGHT:
            return state + 1
        if action == self.LEFT:
            return state - 1
        # この部分は通常到達しない
        raise ValueError(f"無効な行動です: {action}")

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        環境を1ステップ進めます。

        Args:
            action (int): 実行する行動。

        Returns:
            tuple: (次の観測, 報酬, 終了フラグ, 切り捨てフラグ, 情報辞書)
        """
        if self.state is None:
            raise RuntimeError("reset()を呼び出す前にstep()を呼び出すことはできません。")
        
        if self.state == self.exit_pos:
            raise RuntimeError("エピソードは既に終了しています。reset()を呼び出してください。")

        # 行動が有効かチェック
        is_valid_action = self.policy[self.state, action]

        if not is_valid_action:
            if self.throw_invalid_action:
                raise RuntimeError(f"不正な行動です: state={self.state}, action={action}")
            else:
                # 元のPHPコードのロジックに従い、不正な行動で即終了
                terminated = True
                reward = -1.0
                observation = self._get_obs()
                info = {"error": "Unauthorized action"}
                return observation, reward, terminated, False, info
        
        # 状態を更新
        self.state = self._next_state(self.state, action)
        
        # 終了判定
        terminated = (self.state == self.exit_pos)
        
        # 報酬は常に-1.0
        reward = -1.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """環境をレンダリングします"""
        if self.render_mode is None:
            gym.logger.warn(
                "render()を呼び出していますが、render_modeが設定されていません。"
                "環境を初期化する際に `render_mode='human'` または `render_mode='rgb_array'` を指定してください。"
            )
            return None

        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygameがインストールされていません。`pip install pygame` を実行してください。"
            )

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:  # rgb_array
                self.window = pygame.Surface((self.window_size, self.window_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # 背景を白に

        cell_size = self.window_size / max(self.width, self.height)
        line_width = max(1, int(cell_size * 0.05))

        # 壁を描画
        for state in range(self.num_states):
            y, x = divmod(state, self.width)
            
            top_left = (x * cell_size, y * cell_size)
            top_right = ((x + 1) * cell_size, y * cell_size)
            bottom_left = (x * cell_size, (y + 1) * cell_size)
            bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)

            if not self.policy[state, self.UP]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, top_right, line_width)
            if not self.policy[state, self.DOWN]:
                pygame.draw.line(canvas, (0, 0, 0), bottom_left, bottom_right, line_width)
            if not self.policy[state, self.RIGHT]:
                pygame.draw.line(canvas, (0, 0, 0), top_right, bottom_right, line_width)
            if not self.policy[state, self.LEFT]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, bottom_left, line_width)
        
        # ゴールを描画 (赤い円)
        goal_y, goal_x = divmod(self.exit_pos, self.width)
        goal_center = (
            goal_x * cell_size + cell_size * 0.5,
            goal_y * cell_size + cell_size * 0.5,
        )
        pygame.draw.circle(
            canvas, (255, 0, 0), goal_center, cell_size * 0.3
        )

        # エージェントを描画 (青い円)
        if self.state is not None:
            agent_y, agent_x = divmod(self.state, self.width)
            agent_center = (
                agent_x * cell_size + cell_size * 0.5,
                agent_y * cell_size + cell_size * 0.5,
            )
            pygame.draw.circle(
                canvas, (0, 0, 255), agent_center, cell_size * 0.3
            )

        if self.render_mode == "human":
            # ウィンドウに描画内容を転送
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            # フレームレートを維持
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """環境をクリーンアップし、レンダリングウィンドウを閉じます"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
