# import numpy as np
from ..multiagent_mujoco.mujoco_multi import MujocoMulti
from ..custom_suites.utils import tolerance

_STANDUP_HEIGHT = 1.5


class HumanoidStandupMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)
        self.standup_height = kwargs.get("standup_height", _STANDUP_HEIGHT)

        self.tasks = ["standup"]
        self.n_tasks = len(self.tasks)
        self._t = self.wrapped_env.env.env.model.opt.timestep
        self._task_idx = 0

    def step(self, actions):
        raw_reward, done, info = super().step(actions)
        reward = self.get_reward(info)
        return reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        self.wrapped_env.close()

    def _standup_reward(self, info):
        reward_standup = tolerance(info["reward_linup"] * self._t,
                                   bounds=(self.standup_height, float('inf')),
                                   margin=self.standup_height,
                                   value_at_margin=0,
                                   sigmoid="linear")
        return reward_standup

    def get_reward(self, info):
        if self.task == "standup":
            return self._standup_reward(info)
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
