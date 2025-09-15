import numpy as np

from ..multiagent_mujoco.mujoco_multi import MujocoMulti
from ..custom_suites.utils import tolerance

_STAND_HEIGHT = 1.2
_WALK_SPEED = 1
_WALK_BACKWARDS_SPEED = 1
_RUN_SPEED = 8
_RUN_BACKWARDS_SPEED = 6


class Walker2dMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.stand_height = kwargs.get("stand_height", _STAND_HEIGHT)
        self.walk_speed = kwargs.get("hop_speed", _WALK_SPEED)
        self.walk_backwards_speed = kwargs.get("hop_backwards_speed", _WALK_BACKWARDS_SPEED)
        self.run_speed = kwargs.get("spin_speed", _RUN_SPEED)
        self.run_backwards_speed = kwargs.get("spin_backwards_speed", _RUN_BACKWARDS_SPEED)

        body_names = self.wrapped_env.env.env.model.body_names
        self.body_idxes = {name: idx for idx, name in enumerate(body_names)}

        self.tasks = ["stand", "walk", "walk_backwards", "run", "run_backwards"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        raw_reward, done, info = super().step(actions)
        env_actions = np.concatenate(actions)
        info["reward_move"] = raw_reward - 1.0 + 1e-3 * np.square(env_actions).sum()
        info["control"] = env_actions
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _stand_reward(self, info):
        standing = tolerance(self.get_torso_height(),
                             bounds=(self.stand_height, float('inf')),
                             margin=self.stand_height / 2)
        upright = (1 + self.get_torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        return stand_reward

    def _walk_reward(self, info):
        standing = tolerance(self.get_torso_height(),
                             bounds=(self.stand_height, float('inf')),
                             margin=self.stand_height / 2)
        upright = (1 + self.get_torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        move_reward = tolerance(info["reward_move"],
                                bounds=(self.walk_speed, float('inf')),
                                margin=self.walk_speed / 2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        return stand_reward * (5*move_reward + 1) / 6

    def _walk_backwards_reward(self, info):
        standing = tolerance(self.get_torso_height(),
                             bounds=(self.stand_height, float('inf')),
                             margin=self.stand_height / 2)
        upright = (1 + self.get_torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        move_reward = tolerance(-info["reward_move"],
                                bounds=(self.walk_backwards_speed, float('inf')),
                                margin=self.walk_backwards_speed / 2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        return stand_reward * (5*move_reward + 1) / 6

    def _run_reward(self, info):
        standing = tolerance(self.get_torso_height(),
                             bounds=(self.stand_height, float('inf')),
                             margin=self.stand_height / 2)
        upright = (1 + self.get_torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        move_reward = tolerance(info["reward_move"],
                                bounds=(self.run_speed, float('inf')),
                                margin=self.run_speed / 2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        return stand_reward * (5 * move_reward + 1) / 6

    def _run_backwards_reward(self, info):
        standing = tolerance(self.get_torso_height(),
                             bounds=(self.stand_height, float('inf')),
                             margin=self.stand_height / 2)
        upright = (1 + self.get_torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        move_reward = tolerance(-info["reward_move"],
                                bounds=(self.run_backwards_speed, float('inf')),
                                margin=self.run_backwards_speed / 2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        return stand_reward * (5 * move_reward + 1) / 6

    def get_torso_upright(self):
        return self.wrapped_env.env.env.sim.data.body_xmat[self.body_idxes["torso"]][8]  # "zz" projection

    def get_torso_height(self):
        return self.wrapped_env.env.env.sim.data.body_xpos[self.body_idxes["torso"]][2]

    def get_reward(self, info):
        if self.task == "stand":
            return self._stand_reward(info)
        elif self.task == "walk":
            return self._walk_reward(info)
        elif self.task == "walk_backwards":
            return self._walk_backwards_reward(info)
        elif self.task == "run":
            return self._run_reward(info)
        elif self.task == "run_backwards":
            return self._run_backwards_reward(info)
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

    def reset_task(self, task):
        assert task in self.tasks
        self._task_idx = self.tasks.index(task)
        return self.task_idx

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def task(self):
        return self.tasks[self._task_idx]


