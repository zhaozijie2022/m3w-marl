import numpy as np

from ..multiagent_mujoco.mujoco_multi import MujocoMulti
from ..custom_suites.utils import tolerance

_STAND_HEIGHT = 1.4
_WALK_SPEED = 3.0
_RUN_SPEED = 5.0


class HumanoidMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.stand_height = kwargs.get("stand_height", _STAND_HEIGHT)
        self.walk_speed = kwargs.get("walk_speed", _WALK_SPEED)
        self.run_speed = kwargs.get("run_speed", _RUN_SPEED)

        body_names = self.wrapped_env.env.env.model.body_names
        self.body_idxes = {name: idx for idx, name in enumerate(body_names)}

        self.tasks = ["stand", "walk", "run"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        raw_reward, done, info = super().step(actions)
        info["control"] = np.concatenate(actions)
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
        move_reward = tolerance(info["reward_linvel"],
                                bounds=(self.walk_speed, float('inf')),
                                margin=self.walk_speed / 2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        return move_reward

    def _run_reward(self, info):
        move_reward = tolerance(info["reward_linvel"],
                                bounds=(self.run_speed, float('inf')),
                                margin=self.run_speed,
                                value_at_margin=0,
                                sigmoid='linear')

        return move_reward

    def get_torso_upright(self):
        # x_mat: the projection from 1-axes of body to the 2-axes of world
        # xx, xy, xz, yx, yy, yz, zx, zy, zz
        return self.wrapped_env.env.env.sim.data.body_xmat[self.body_idxes["torso"]][8]

    def get_torso_height(self):
        return self.wrapped_env.env.env.sim.data.body_xpos[self.body_idxes["torso"]][2]

    def get_center_of_mass_velocity(self):
        return self.wrapped_env.env.env.sim.data.subtree_linvel[self.body_idxes["torso"]].copy()

    def get_reward(self, info):
        if self.task == "stand":
            return self._stand_reward(info)
        elif self.task == "walk":
            return self._walk_reward(info)
        elif self.task == "run":
            return self._run_reward(info)
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

