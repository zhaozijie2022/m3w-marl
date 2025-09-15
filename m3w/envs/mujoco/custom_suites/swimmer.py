from ..multiagent_mujoco.mujoco_multi import MujocoMulti
from ..custom_suites.utils import tolerance
_SWIM_SPEED = 1
_SWIM_BACKWARDS_SPEED = 0.8


class SwimmerMulti(MujocoMulti):
    def __init__(self, env_args, **kwargs):
        super().__init__(env_args=env_args, **kwargs)
        self.swim_speed = kwargs.get("swim_speed", _SWIM_SPEED)
        self.swim_backwards_speed = kwargs.get("swim_backwards_speed", _SWIM_BACKWARDS_SPEED)

        self.tasks = ["swim", "swim_backwards"]
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

    def _swim_reward(self, info):
        speed = info["reward_fwd"]
        return tolerance(speed,
                         bounds=(self.swim_speed, float('inf')),
                         margin=self.swim_speed,
                         value_at_margin=0,
                         sigmoid='linear')

    def _swim_backwards_reward(self, info):
        speed = -1.0 * info["reward_fwd"]
        return tolerance(speed,
                         bounds=(self.swim_backwards_speed, float('inf')),
                         margin=self.swim_backwards_speed,
                         value_at_margin=0,
                         sigmoid='linear')

    def get_reward(self, info):
        if self.task == "swim":
            return self._swim_reward(info)
        elif self.task == "swim_backwards":
            return self._swim_backwards_reward(info)
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
