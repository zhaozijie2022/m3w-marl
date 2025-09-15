import numpy as np
from gym.spaces import Box
import torch
from envs.wrappers import ShareSubprocVecEnv


class MultitaskDexHands:
    def __init__(self, env_args, env_fns=None):
        self.env_args = env_args
        self.envs = ShareSubprocVecEnv(env_fns)

        for k in range(self.env_args["n_tasks"]):
            self.envs.remotes[k].send(("get_spaces", None))

        results = [remote.recv() for remote in self.envs.remotes]
        obs_space, share_obs_space, act_space = zip(*results)

        self.obs_size = max([obs_space[i][0].shape[0] for i in range(len(obs_space))])
        self.share_obs_size = max([share_obs_space[i][0].shape[0] for i in range(len(share_obs_space))])
        self.action_shape = max([act_space[i][0].shape[0] for i in range(len(act_space))])

        self.n_agents = 2
        #
        self.observation_space = [Box(low=np.array([-10] * self.obs_size, dtype=np.float32),
                                      high=np.array([10] * self.obs_size, dtype=np.float32),
                                      dtype=np.float32) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=np.array([-10] * self.share_obs_size, dtype=np.float32),
                                            high=np.array([10] * self.share_obs_size, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = tuple([Box(low=np.array([-1] * self.action_shape, dtype=np.float32),
                                       high=np.array([1] * self.action_shape, dtype=np.float32),
                                       dtype=np.float32) for _ in range(self.n_agents)])

    def reset(self):
        for remote in self.envs.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.envs.remotes]  #[n_tasks, (obs, share_obs, available_actions)]
        obs, share_obs, available_actions = zip(*results)
        obs_n = torch.concat(obs)
        share_obs_n = torch.concat(share_obs)
        available_actions = None

        return obs_n, share_obs_n, available_actions

    def meta_reset_task(self, task_idxes):
        r_tasks = []
        for i in range(self.env_args["n_tasks"]):
            r_tasks.extend([i] * self.env_args["n_envs_per_task"])
        return r_tasks

    def step(self, actions):
        # [Nt * Ne, Na, dim]
        actions = actions.reshape(self.env_args["n_tasks"], self.env_args["n_envs_per_task"], self.n_agents, self.action_shape)
        actions = actions.transpose(1, 2)
        for k in range(self.env_args["n_tasks"]):
            self.envs.remotes[k].send(('step', actions[k]))
        # for remote, action in zip(self.envs.remotes, actions):
        #     remote.send(('step', action))
        self.envs.waiting = True
        results = [remote.recv() for remote in self.envs.remotes]
        self.envs.waiting = False

        obs, share_obs, rews, dones, infos, available_actions = zip(*results)

        obs_n = torch.concat(obs)
        share_obs_n = torch.concat(share_obs)
        reward_n = torch.concat(rews)
        done_n = torch.concat(dones)
        info_n = []
        available_actions = None
        return obs_n, share_obs_n, reward_n, done_n, info_n, available_actions

    def close(self):
        for env in self.envs:
            env.close()

