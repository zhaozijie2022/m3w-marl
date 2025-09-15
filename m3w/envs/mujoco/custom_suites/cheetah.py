from ..multiagent_mujoco.mujoco_multi import MujocoMulti
from ..custom_suites.utils import tolerance
import numpy as np

_RUN_SPEED = 10
_RUN_BACKWARDS_SPEED = 8
_RUN_ONE_FOOT_SPEED = 6
_JUMP_SPEED = 0.5
_JUMP_HEIGHT = 1.2


class HalfCheetahMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)

        self.run_speed = kwargs.get("run_speed", _RUN_SPEED)
        self.run_backwards_speed = kwargs.get("run_backwards_speed", _RUN_BACKWARDS_SPEED)
        self.run_one_foot_speed = kwargs.get("run_one_foot_speed", _RUN_ONE_FOOT_SPEED)
        self.jump_speed = kwargs.get("jump_speed", _JUMP_SPEED)
        self.jump_height = kwargs.get("jump_height", _JUMP_HEIGHT)

        body_names = self.wrapped_env.env.env.model.body_names
        self.body_idxes = {name: idx for idx, name in enumerate(body_names)}
        self.xyz_index = {'x': 0, 'y': 1, 'z': 2}

        self.tasks = ["run", "run_backwards", "jump", "run_front", "run_back"]
        self.n_tasks = len(self.tasks)
        self._task_idx = 0

    def step(self, actions):
        reward, done, info = super().step(actions)
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _run_reward(self, info):
        speed = info["reward_run"]
        return tolerance(speed,
                         bounds=(self.run_speed, float('inf')),
                         margin=self.run_speed,
                         value_at_margin=0,
                         sigmoid='linear')

    def _run_backwards_reward(self, info):
        speed = -1.0 * info["reward_run"]
        return tolerance(speed,
                         bounds=(self.run_backwards_speed, float('inf')),
                         margin=self.run_backwards_speed,
                         value_at_margin=0,
                         sigmoid='linear')

    def _jump_reward(self, info):
        front_reward = self._stand_one_foot_reward(info, "ffoot")
        back_reward = self._stand_one_foot_reward(info, "bfoot")
        return 0.5 * (front_reward + back_reward)

    def _run_front_reward(self, info):
        return self._run_one_foot_reward(info, "bfoot")

    def _run_back_reward(self, info):
        return self._run_one_foot_reward(info, "ffoot")

    def _stand_one_foot_reward(self, info, which_foot):
        speed = info["reward_run"]
        speed_reward = tolerance(speed,
                                 bounds=(-self.jump_speed, +self.jump_speed),
                                 margin=self.jump_speed,
                                 value_at_margin=0,
                                 sigmoid='linear')

        torso_height = self.get_body_pos("torso", 'z')
        foot_height = self.get_body_pos(which_foot, 'z')
        height = 0.5 * (torso_height + foot_height)
        height_reward = tolerance(height,
                                  bounds=(self.jump_height, float('inf')),
                                  margin=0.5 * self.jump_height)

        return (5 * height_reward + speed_reward) / 6

    def _run_one_foot_reward(self, info, which_foot):
        torso_height = self.get_body_pos("torso", 'z')
        torso_up = tolerance(torso_height,
                             bounds=(self.jump_height, float('inf')),
                             margin=0.5 * self.jump_height)
        foot_height = self.get_body_pos(which_foot, 'z')
        foot_up = tolerance(foot_height,
                            bounds=(self.jump_height, float('inf')),
                            margin=0.5 * self.jump_height)
        up_reward = (3 * foot_up + 2 * torso_up) / 5

        speed = info["reward_run"]
        speed_reward = tolerance(speed,
                                 bounds=(self.run_one_foot_speed, float('inf')),
                                 margin=self.run_one_foot_speed, )
        return up_reward * (5 * speed_reward + 1) / 6

    def get_body_pos(self, name, xyz):
        return self.wrapped_env.env.env.sim.data.body_xpos[self.body_idxes[name]][self.xyz_index[xyz]]

    def get_reward(self, info):
        if self.task == "run":
            return self._run_reward(info)
        elif self.task == "run_backwards":
            return self._run_backwards_reward(info)
        elif self.task == "jump":
            return self._jump_reward(info)
        elif self.task == "run_front":
            return self._run_front_reward(info)
        elif self.task == "run_back":
            return self._run_back_reward(info)
        else:
            raise NotImplementedError(f"task {self.task} is not implemented.")

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

