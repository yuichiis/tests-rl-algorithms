import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

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

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.state: Optional[int] = None
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

    def _get_obs(self) -> int:
        return self.state

    def _get_info(self) -> Dict[str, Any]:
        valid_actions_flags = self.policy[self.state]
        valid_actions = [i for i, v in enumerate(valid_actions_flags) if v]
        return {"valid_actions": valid_actions}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _next_state(self, state: int, action: int) -> int:
        if action == self.UP: return state - self.width
        if action == self.DOWN: return state + self.width
        if action == self.RIGHT: return state + 1
        if action == self.LEFT: return state - 1
        raise ValueError(f"無効な行動です: {action}")

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if self.state is None: raise RuntimeError("reset()を呼び出す前にstep()を呼び出すことはできません。")
        if self.state == self.exit_pos: raise RuntimeError("エピソードは既に終了しています。reset()を呼び出してください。")

        is_valid_action = self.policy[self.state, action]
        if not is_valid_action:
            if self.throw_invalid_action:
                raise RuntimeError(f"不正な行動です: state={self.state}, action={action}")
            else:
                return self._get_obs(), -1.0, True, False, {"error": "Unauthorized action"}
        
        self.state = self._next_state(self.state, action)
        terminated = (self.state == self.exit_pos)
        reward = -1.0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            gym.logger.warn("render()を呼び出していますが、render_modeが設定されていません。")
            return None
        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled("pygameがインストールされていません。`pip install pygame` を実行してください。")

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # ### 修正点 ###
        # 単一のcell_sizeではなく、幅と高さで別々のセルサイズを計算する
        cell_width = self.window_size / self.width
        cell_height = self.window_size / self.height
        line_width = max(1, int(min(cell_width, cell_height) * 0.05))

        # 壁を描画
        for state in range(self.num_states):
            y, x = divmod(state, self.width)

            
            # ### 修正点 ###
            # cell_widthとcell_heightを使って各コーナーの座標を正確に計算する
            top_left = (x * cell_width, y * cell_height)
            top_right = ((x + 1) * cell_width, y * cell_height)
            bottom_left = (x * cell_width, (y + 1) * cell_height)
            bottom_right = ((x + 1) * cell_width, (y + 1) * cell_height)


            if not self.policy[state, self.UP]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, top_right, line_width)
            if not self.policy[state, self.DOWN]:
                pygame.draw.line(canvas, (0, 0, 0), bottom_left, bottom_right, line_width)
            if not self.policy[state, self.RIGHT]:
                pygame.draw.line(canvas, (0, 0, 0), top_right, bottom_right, line_width)
            if not self.policy[state, self.LEFT]:
                pygame.draw.line(canvas, (0, 0, 0), top_left, bottom_left, line_width)
        
        # ゴールを描画
        goal_y, goal_x = divmod(self.exit_pos, self.width)
        goal_center = (
            goal_x * cell_width + cell_width * 0.5,
            goal_y * cell_height + cell_height * 0.5,
        )
        pygame.draw.circle(
            canvas, (255, 0, 0), goal_center, min(cell_width, cell_height) * 0.3
        )

        # エージェントを描画
        if self.state is not None:
            agent_y, agent_x = divmod(self.state, self.width)
            agent_center = (
                agent_x * cell_width + cell_width * 0.5,
                agent_y * cell_height + cell_height * 0.5,
            )
            pygame.draw.circle(
                canvas, (0, 0, 255), agent_center, min(cell_width, cell_height) * 0.3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None