import numpy as np
from gym.spaces import Box
from .custom_suites import ENV_REGISTRY, ARGS_REGISTRY


class MultitaskMujoco:
    def __init__(self, env_args):
        self.env_args = env_args
        self.envs = self.make_envs()
        self.n_tasks = sum([len(tasks) for tasks in env_args["envs"].values()])
        self.envs_info = [env.get_env_info() for env in self.envs]

        self.obs_size = max([env_info["obs_shape"] for env_info in self.envs_info]) + self.n_tasks
        self.share_obs_size = 1
        self.action_shape = max([env_info["n_actions"] for env_info in self.envs_info])
        self.mt_action_shape_n = self.get_action_shape_n()  # List[List[]*n_agents]*n_domains

        self.agent_nums = [env_info["n_agents"] for env_info in self.envs_info]

        self.n_agents = max(self.agent_nums)

        self.observation_space = [Box(low=np.array([-10] * self.obs_size, dtype=np.float32),
                                      high=np.array([10] * self.obs_size, dtype=np.float32),
                                      dtype=np.float32) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=np.array([-10] * self.share_obs_size, dtype=np.float32),
                                            high=np.array([10] * self.share_obs_size, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = tuple([Box(low=np.array([-1] * self.action_shape, dtype=np.float32),
                                       high=np.array([1] * self.action_shape, dtype=np.float32),
                                       dtype=np.float32) for _ in range(self.n_agents)])

        self.tasks, self.domain_belong, self.in_domain_idx = [], [], []
        
        for i, tasks in enumerate(env_args["envs"].values()):
            self.tasks.extend(tasks)
            self.domain_belong.extend([i] * len(tasks))
            self.in_domain_idx.extend(list(range(len(tasks))))

        self._task_idx = 0

    def reset(self):
        obs_n = self._obs_pat(self.env.reset())
        share_obs_n = [np.zeros(1) for _ in range(len(obs_n))]
        # share_obs_n = [self._state_pat(self.env.get_state()) for _ in range(len(obs_n))]
        available_actions = [None] * len(obs_n)
        return obs_n, share_obs_n, available_actions

    def step(self, actions):
        actions_env = self._act_crop(actions)

        reward, done, info = self.env.step(actions_env)

        obs_n = self._obs_pat(self.env.get_obs())
        share_obs_n = [np.zeros(1) for _ in range(len(obs_n))]
        # share_obs_n = [self._state_pat(self.env.get_state()) for _ in range(len(obs_n))]

        reward_n = [np.array([reward]) for _ in range(len(obs_n))]
        done_n = [done for _ in range(len(obs_n))]
        info["task"] = self.task
        info["task_idx"] = self.task_idx
        info["bad_transition"] = False
        info_n = [info for _ in range(len(obs_n))]

        available_actions = [None] * len(obs_n)
        return obs_n, share_obs_n, reward_n, done_n, info_n, available_actions

    def reset_task(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(len(self.tasks))
        self._task_idx = task_idx
        self.env.reset_task(self.task)
        return self.task_idx

    def close(self):
        for env in self.envs:
            env.close()

    def make_envs(self):
        envs = []
        for domain, tasks in self.env_args["envs"].items():
            env_args = ARGS_REGISTRY[domain]
            env_args["episode_limit"] = self.env_args["episode_limit"]
            env = ENV_REGISTRY[domain](env_args=env_args)
            envs.append(env)
        return envs

    def get_action_shape_n(self):
        mt_action_shape_n = []
        for env_info in self.envs_info:
            mt_action_shape_n.append([])
            for action_space in env_info["action_spaces"]:
                mt_action_shape_n[-1].append(action_space.shape[0])
        return mt_action_shape_n

    def get_env_names(self):
        env_names = []
        for domain, tasks in self.env_args["envs"].items():
            for task in tasks:
                env_names.append(f"{domain}_{task}")
        return env_names

    def get_action_mask(self):
        action_mask = np.zeros((self.n_tasks, self.n_agents, self.action_shape))
        _t = 0
        for i_d, domain in enumerate(self.env_args["envs"].keys()):
            for i_t, task in enumerate(self.env_args["envs"][domain]):
                for i_a in range(self.agent_nums[i_d]):
                    action_mask[_t, i_a, :self.mt_action_shape_n[i_d][i_a]] = 1.0
                _t += 1
        return action_mask

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)

    @property
    def task(self):
        return self.tasks[self.task_idx]

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def env_idx(self):
        return self.domain_belong[self.task_idx]

    @property
    def env(self):
        return self.envs[self.env_idx]

    @property
    def steps(self):
        return self.envs[self.env_idx].steps

    def _act_crop(self, actions):
        actions_env = [
            actions[i][:self.mt_action_shape_n[self.env_idx][i]]
            for i in range(self.agent_nums[self.env_idx])
        ]
        return actions_env

    def _obs_pat(self, obs_n):
        onehot_task = np.zeros(self.n_tasks)
        onehot_task[self.task_idx] = 1.0
        for i, obs in enumerate(obs_n):
            obs_n[i] = np.concatenate([onehot_task, obs, np.zeros(self.obs_size - len(obs) - self.n_tasks)])
        for _ in range(len(obs_n), self.n_agents):
            obs_n.append(np.zeros(self.obs_size))
        return obs_n

    def _state_pat(self, state):
        onehot_task = np.zeros(self.n_tasks)
        onehot_task[self.task_idx] = 1.0
        return np.concatenate([onehot_task, state, np.zeros(self.share_obs_size - len(state) - self.n_tasks)])

    def render(self, **kwargs):
        return self.env.render(**kwargs)

