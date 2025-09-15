import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import setproctitle
from m3w.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    set_seed,
    get_num_agents,
)
from m3w.utils.models_tools import init_device
from m3w.utils.configs_tools import save_config, get_task_name
from m3w.algorithms.actors.world_model_actor import WorldModelActor
from m3w.algorithms.critics.world_model_critic import WorldModelCritic
from m3w.common.buffers.world_model_buffer import WorldModelBuffer

from gym.spaces import Box
# from tqdm import tqdm
import wandb
import m3w.models.world_models as wms
from m3w.utils.envs_tools import check
import datetime
import itertools

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def convert_num(num):
    suffixes = ['K', 'M', 'G', 'T']
    for suffix in suffixes:
        num /= 1000
        if num < 1000:
            return f'{num:.1f}{suffix}'
    return f'{num:.3f}T'


class WorldModelRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        assert args["algo"] == "m3w"
        assert env_args.get("state_type", "EP") == "EP"
        assert algo_args["render"]["use_render"] is False

        # read config
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.policy_freq = self.algo_args["algo"].get("policy_freq", 1)
        self.state_type = env_args.get("state_type", "EP")
        self.fixed_order = algo_args["algo"]["fixed_order"]

        self.num_pi_trajs = algo_args["plan"]["num_pi_trajs"]
        self.num_samples = algo_args["plan"]["num_samples"]
        self.num_elites = algo_args["plan"]["num_elites"]
        self.plan_iter = algo_args["plan"]["iterations"]
        self.horizon = algo_args["plan"]["horizon"]
        self.max_std = algo_args["plan"]["max_std"]
        self.min_std = algo_args["plan"]["min_std"]
        self.temperature = algo_args["plan"]["temperature"]
        self.entropy_coef = algo_args["model"]["entropy_coef"]
        self.latent_dim = algo_args["world_model"]["latent_dim"]
        self.step_rho = algo_args["world_model"]["step_rho"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.run_dir, self.save_dir, self.log_file, self.task_name, self.expt_name = self.init_config()

        if algo_args["logger"].get("wandb", False):
            wandb.init(
                project="hawm_" + self.task_name,
                group=args["algo"],
                name=self.expt_name,
                config={"main_args": args, "algo_args": algo_args, "env_args": env_args},
            )
        setproctitle.setproctitle("what-can-I-say?-mamba-out")

        # region make env
        assert algo_args["train"]["n_rollout_threads"] % env_args["n_tasks"] == 0
        num_env_per_task = algo_args["train"]["n_rollout_threads"] // env_args["n_tasks"]
        self.task_idxes = list(np.arange(env_args["n_tasks"]).repeat(num_env_per_task))
        self.envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            algo_args["train"]["n_rollout_threads"],
            env_args,
        )
        self.envs.meta_reset_task(self.task_idxes)
        if algo_args["eval"]["use_eval"]:
            assert algo_args["eval"]["n_eval_rollout_threads"] == algo_args["train"][
                "n_rollout_threads"], "equal for running_mean in planner"
            self.eval_envs = make_eval_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["eval"]["n_eval_rollout_threads"],
                env_args,
            )
            self.eval_envs.meta_reset_task(self.task_idxes)
        else:
            self.eval_envs = None
        
        self.num_tasks = env_args["n_tasks"]
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros((self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1))
        self.action_spaces = self.envs.action_space
        self.action_dims = []
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)
            self.action_dims.append(self.action_spaces[agent_id].shape[0])
        print("init env, done")
        # endregion

        # region decentralized dynamics & reward model
        action_sum_dim = sum(self.action_dims)
        self.obs_encoder = []
        for agent_id in range(self.num_agents):
            _obs_encoder = wms.MLPEncoder(
                in_dim=self.envs.observation_space[agent_id].shape[0],
                mlp_dims=[algo_args["world_model"]["latent_dim"]] * max(algo_args["world_model"]["num_enc_layers"] - 1, 1),
                out_dim=algo_args["world_model"]["latent_dim"],
                act=wms.SimNorm(algo_args["world_model"]["simnorm_dim"]),
                # lr=algo_args["model"]["lr"] * algo_args["world_model"]["enc_lr_scale"],
                device=self.device,
            )
            self.obs_encoder.append(_obs_encoder)

        # self.dynamics_model = wms.CenMLPDynamicsModel(
        #     in_dim=algo_args["world_model"]["latent_dim"] * self.num_agents + action_sum_dim,
        #     mlp_dims=[algo_args["world_model"]["latent_dim"] * self.num_agents + action_sum_dim] * 2,
        #     out_dim=algo_args["world_model"]["latent_dim"] * self.num_agents,
        #     act=wms.SimNorm(algo_args["world_model"]["simnorm_dim"]),
        #     device=self.device,
        # )

        # self.reward_model = wms.CenMLPRewardModel(
        #     in_dim=algo_args["world_model"]["latent_dim"] * self.num_agents + action_sum_dim,
        #     mlp_dims=[algo_args["world_model"]["latent_dim"] * self.num_agents + action_sum_dim] * 2,
        #     out_dim=max(algo_args["world_model"]["num_bins"], 1),
        #     act=None if algo_args["world_model"]["num_bins"] else F.sigmoid,
        #     device=self.device,
        # )
        # self.reward_model.mlp[-1].weight.data.fill_(0)
        
        # MLP
        self.dynamics_model = wms.CenMoEDynamicsModel(
            d_z=algo_args["world_model"]["latent_dim"],
            d_a=self.action_dims[0],
            mlp_dims=[512, 512],
            n_experts=algo_args["world_model"]["num_dynamics_experts"],
            act=wms.SimNorm(algo_args["world_model"]["simnorm_dim"]),
            device=self.device,
        )
        
        self.reward_model = wms.CenMoERewardModel(
            d_z=algo_args["world_model"]["latent_dim"],
            d_a=self.action_dims[0],
            n_agents=self.num_agents,
            n_experts=algo_args["world_model"]["num_reward_experts"],
            k=algo_args["world_model"]["top_k"],
            n_r=max(algo_args["world_model"]["num_bins"], 1),
            n_heads=1,
            device=self.device,
        )

        self.reward_processor = wms.TwoHotProcessor(
            num_bins=algo_args["world_model"]["num_bins"],
            vmin=algo_args["world_model"]["reward_min"],
            vmax=algo_args["world_model"]["reward_max"],
            device=self.device,
        )
        # endregion

        # region actor, critic, buffer, actor, critic
        self.latent_space = Box(low=-10, high=10, shape=(algo_args["world_model"]["latent_dim"],), dtype=np.float32)
        self.joint_latent_space = Box(low=-10, high=10, shape=(algo_args["world_model"]["latent_dim"] * self.num_agents,), dtype=np.float32)

        self.actor = []
        for agent_id in range(self.num_agents):
            agent = WorldModelActor(
                {**algo_args["model"], **algo_args["algo"]},
                self.latent_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.actor.append(agent)
        self.critic = WorldModelCritic(
            args={**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            share_obs_space=self.joint_latent_space,
            act_space=self.envs.action_space,
            num_agents=self.num_agents,
            state_type=self.state_type,
            device=self.device,
            wm_args=algo_args["world_model"]
        )
        self.buffer = WorldModelBuffer(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"], **env_args},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()
            print("restore model, done")
        else:
            print("init actor, critic, buffer, done")
        # endregion

        # optimizer
        _params = []
        for agent_id in range(self.num_agents):
            _params.append({
                'params': self.obs_encoder[agent_id].parameters(),
                'lr': algo_args["model"]["lr"] * algo_args["world_model"]["enc_lr_scale"],
            })
        _params.append({
            'params': self.dynamics_model.parameters(),
        })
        _params.append({
            'params': self.reward_model.parameters(),
        })
        _params.append({
            'params': itertools.chain(
                self.critic.critic.parameters(),
                self.critic.critic2.parameters()
            ),
            'lr': algo_args["model"]["lr"]
        })
        self.model_optimizer = torch.optim.Adam(
            params=_params,
            lr=algo_args["model"]["lr"],
        )

        # region init training params
        self.total_it = 0  # total iteration
        self._start_time = self._check_time = time.time()
        self.train_episode_rewards, self.done_episodes_rewards = None, None
        self.plan_sample_errors = None
        self.running_mean = None
        self.t0 = None
        # endregion

    def run(self):
        self._start_time = self._check_time = time.time()

        steps = self.algo_args["train"]["num_env_steps"] // self.algo_args["train"]["n_rollout_threads"]
        update_num = int(self.algo_args["train"]["update_per_train"] * self.algo_args["train"]["train_interval"])

        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = [[] for _ in range(self.num_tasks)]
        self.running_mean = [
            torch.zeros(
                self.horizon,
                self.algo_args["train"]["n_rollout_threads"],
                self.action_dims[agent_id]
            ).to(**self.tpdv) for agent_id in range(self.num_agents)
        ]
        self.t0 = [True] * self.algo_args["train"]["n_rollout_threads"]

        # warmup
        if self.algo_args["train"]["model_dir"] is not None:
            obs, share_obs, available_actions = self.envs.reset()
        else:
            print("start warmup")
            obs, share_obs, available_actions = self.warmup(
                warmup_train=self.algo_args["world_model"]["warmup_train"],
                train_steps=self.algo_args["world_model"]["wt_steps"],
            )
            print("finish warmup, start training")

        train_info = {}
        for step in range(1, steps + 1):
            # region rollout
            if True:
                actions = self.plan(
                    obs=obs,
                    t0=self.t0,
                    available_actions=None,
                    add_random=True
                )
            else:
                actions = self.get_actions(
                    obs=obs,
                    available_actions=None,
                    add_random=True
                )
            (new_obs, new_share_obs, rewards,
             dones, infos, _,) = self.envs.step(actions)
            # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
            # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)

            # if True:
            #     _d_err, _r_err = self.test_error(obs, actions, rewards, new_obs)
            #     self.plan_sample_errors[0].append(_d_err)
            #     self.plan_sample_errors[1].append(_r_err)

            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            # endregion

            # train
            if step % self.algo_args["train"]["train_interval"] == 0:
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)

                for update_it in range(update_num):
                    _info = self.train()
                    for k, v in _info.items():
                        if k not in train_info:
                            train_info[k] = []
                        train_info[k].append(v)

            # log train_info
            if step % self.algo_args["train"]["log_interval"] == 0:
                for k, v in train_info.items():
                    train_info[k] = np.mean(v)

                rollout_info = {
                    "rew_buffer": self.buffer.get_mean_rewards(),
                }
                for task_id in range(self.num_tasks):
                    if len(self.done_episodes_rewards[task_id]) > 0:
                        rollout_info[f"r_{task_id}"] = np.mean(self.done_episodes_rewards[task_id])
                        self.done_episodes_rewards[task_id] = []

                self.console_log(
                    step,
                    train_info=train_info,
                    rollout_info=rollout_info,
                )
                train_info = {}

            # eval
            if step % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    print(f"\nEvaluation at step {step} / {steps}:")
                    self.eval(step, use_plan=self.algo_args["plan"]["use_plan"])

            # save
            if step % self.algo_args["train"]["save_interval"] == 0 and self.algo_args["logger"]["save_model"]:
                self.save()
                print("\nModel has been saved\n")

    def warmup(self, warmup_train=False, train_steps=10_000):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, _ = self.envs.reset()
        for _ws in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(None)
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                _,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs

        if warmup_train:
            warmup_info = {}
            for _i in range(1, train_steps + 1):
                _info = self.train(train_actor=True)
                for k, v in _info.items():
                    if k not in warmup_info:
                        warmup_info[k] = []
                    warmup_info[k].append(v)

                if _i % 100 == 0:
                    for k, v in warmup_info.items():
                        warmup_info[k] = np.mean(v)
                    self.console_log(
                        _step=self.total_it,
                        warmup_info=warmup_info,
                    )
                    # print(_i, "warmup-train" + "".join([", %s: %.6f" % (k, v) for k, v in warmup_info.items()]))
                    warmup_info = {}

        return obs, share_obs, None

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: # list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths
        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        terms = np.full((self.algo_args["train"]["n_rollout_threads"], 1), False)
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                ):
                    terms[i][0] = True

        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards[self.task_idxes[i]].append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                self.t0[i] = True
            else:
                self.t0[i] = False

        data = (
            share_obs[:, 0],  # (n_threads, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards[:, 0],  # (n_threads, 1)
            np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
            valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
            terms,  # (n_threads, 1)
            next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
            next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        )

        self.buffer.insert(data)

    def sample_actions(self, available_actions=None):
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.algo_args["train"]["n_rollout_threads"]):
                action.append(self.action_spaces[agent_id].sample())
            actions.append(action)
        return np.array(actions).transpose(1, 0, 2)

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        actions = []
        for agent_id in range(self.num_agents):
            _obs = check(obs[:, agent_id]).to(**self.tpdv)
            _z = self.obs_encoder[agent_id].encode(_obs)
            actions.append(
                self.actor[agent_id].get_actions(
                    _z, stochastic=add_random
                ).detach().cpu().numpy()
            )
        return np.array(actions).transpose(1, 0, 2)
    
    @torch.no_grad()
    def plan(self, obs, t0=None, available_actions=None, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, (n_threads, n_agents, dim)
            t0: (bool)
            available_actions: (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        n_threads = obs.shape[0]
        gamma = self.algo_args["algo"]["gamma"]
        t0 = [True] * n_threads if t0 is None else t0
        # data dim [n_agents, (horizon, n_threads, num_samples, *dim)]

        # zs.shape [n_agents, (n_threads, dim)]
        zs = [
            self.obs_encoder[agent_id].encode(check(obs[:, agent_id]).to(**self.tpdv))
            for agent_id in range(self.num_agents)
        ]

        # act_mean/act_std.shape: [n_agents, (horizon, n_threads, dim)]
        act_mean = [torch.zeros(
            self.horizon,
            n_threads,
            self.action_dims[agent_id]
        ).to(**self.tpdv) for agent_id in range(self.num_agents)
                    ]
        act_std = [
            self.max_std * torch.ones_like(act_mean[agent_id]).to(**self.tpdv)
            for agent_id in range(self.num_agents)
        ]

        for thread in range(n_threads):
            if not t0[thread]:
                for agent_id in range(self.num_agents):
                    act_mean[agent_id][:-1, thread] = self.running_mean[agent_id][1:, thread]

        # actions.shape: [n_agents, (horizon, n_threads, num_samples, dim)]
        actions = [
            torch.zeros(
                self.horizon,
                n_threads,
                self.num_samples,
                self.action_dims[agent_id]
            ).to(**self.tpdv) for agent_id in range(self.num_agents)
        ]


        if self.num_pi_trajs > 0:
            # pi_actions.shape: [n_agents, (horizon, n_threads, num_pi_trajs, dim)]
            pi_actions = [
                torch.zeros(
                    self.horizon,
                    n_threads,
                    self.num_pi_trajs,
                    self.action_dims[agent_id]
                ).to(**self.tpdv) for agent_id in range(self.num_agents)
            ]
            # _zs.shape: [n_agents, (n_threads, num_pi_trajs, dim)]
            _zs = torch.stack([
                zs[agent_id].unsqueeze(1).repeat(1, self.num_pi_trajs, 1)
                for agent_id in range(self.num_agents)
            ], dim=0)  # _zs: (n_agents, n_threads, num_pi_trajs, dim)
            # world-model+policy
            for t in range(self.horizon - 1):
                for agent_id in range(self.num_agents):
                    pi_actions[agent_id][t] = self.actor[agent_id].get_actions(
                        obs=_zs[agent_id],
                        available_actions=None,
                        stochastic=True,
                    )

                joint_zs_t = _zs.permute(1, 2, 0, 3).reshape(n_threads * self.num_pi_trajs, self.num_agents, -1)  # (nt, np, na, d) -> (nt*np, na, d)
                joint_pi_actions_t = torch.stack(
                    [pi_actions[agent_id][t] for agent_id in range(self.num_agents)],
                    dim=-2
                ).reshape(n_threads * self.num_pi_trajs, self.num_agents, -1)  # (nt, np, na, d) -> (nt*np, na*d)
                _zs = self.dynamics_model.predict(
                    z=joint_zs_t,
                    a=joint_pi_actions_t
                ).reshape(n_threads, self.num_pi_trajs, self.num_agents, -1)
                _zs = _zs.permute(2, 0, 1, 3).contiguous()  # (n_agents, n_threads, num_pi_trajs, dim)

            for agent_id in range(self.num_agents):
                pi_actions[agent_id][-1] = self.actor[agent_id].get_actions(
                    obs=_zs[agent_id],
                    available_actions=None,
                    stochastic=True
                )
                actions[agent_id][:, :, :self.num_pi_trajs, :] = pi_actions[agent_id].clone()
                # (horizon, n_threads, num_samples, action_dim)

        # out_a.shape: [n_agents, (n_threads, dim)]
        out_a = [
            torch.zeros((
                n_threads,
                self.action_dims[agent_id]
            )).to(**self.tpdv) for agent_id in range(self.num_agents)
        ]

        # plan-iter
        for _i in range(self.plan_iter):
            _zs = [
                zs[agent_id].unsqueeze(1).repeat(1, self.num_samples, 1)
                for agent_id in range(self.num_agents)
            ]
            
            for agent_id in range(self.num_agents):
                actions[agent_id][:, :, self.num_pi_trajs:] = torch.normal(
                    mean=act_mean[agent_id].unsqueeze(2).repeat(1, 1, self.num_samples - self.num_pi_trajs, 1),
                    std=act_std[agent_id].unsqueeze(2).repeat(1, 1, self.num_samples - self.num_pi_trajs, 1)
                ).clamp(-1, 1)

            # g_returns.shape: [n_agents, (n_threads, num_samples, 1)]
            g_returns = self.estimate_value(
                _zs, actions, gamma
            )

            for agent_id in range(self.num_agents):

                # value.shape: (n_threads, num_samples)
                value = torch.mean(torch.stack(g_returns, dim=0), dim=0).squeeze(-1)
                # value = g_returns[agent_id].squeeze(-1)

                # idxes.shape: (n_threads, num_elites)
                elite_idxes = torch.topk(value, self.num_elites, dim=-1)[1]
                elite_values = torch.gather(
                    value, dim=-1,  # value.shape: (n_threads, num_samples)
                    index=elite_idxes  # idxes.shape: (n_threads, num_elites)
                )
                elite_actions = torch.gather(
                    actions[agent_id], dim=2,  # actions[agent_id].shape: (horizon, n_threads, num_samples, action_dim)
                    index=elite_idxes.unsqueeze(0).unsqueeze(-1).repeat(self.horizon, 1, 1, self.action_dims[agent_id])
                    # index.shape: (horizon, n_threads, num_elites, action_dim)
                )

                # score.shape: (n_threads, num_elites) -> (1, n_threads, num_elites, 1)
                max_value = elite_values.max(dim=-1, keepdim=True)[0]
                score = torch.exp(self.temperature * (elite_values - max_value))
                score = score / score.sum(dim=-1, keepdim=True)
                score = score.unsqueeze(0).unsqueeze(-1)

                act_mean[agent_id] = (
                        score * elite_actions
                ).sum(dim=2)
                act_std[agent_id] = (
                        score * (elite_actions - act_mean[agent_id].unsqueeze(2)) ** 2
                ).sum(dim=2)
                act_std[agent_id] = torch.sqrt(act_std[agent_id] + 1e-6).clamp_(self.min_std, self.max_std)  # (n_threads, action_dim)

                if _i == self.plan_iter - 1:
                    # score.shape: (n_threads, num_elites)
                    score = score.squeeze(0).squeeze(-1).cpu().numpy()
                    for thread in range(n_threads):
                        idx = np.random.choice(np.arange(self.num_elites), p=score[thread][:])
                        out_a[agent_id][thread] = elite_actions[0, thread, idx, :]
                        if add_random:
                            out_a[agent_id][thread] += torch.randn_like(out_a[agent_id][thread]) * act_std[agent_id][0, thread]
                            out_a[agent_id][thread] = out_a[agent_id][thread].clamp(-1, 1)

        self.running_mean = act_mean
        out_a = [out_a[agent_id].cpu().numpy() for agent_id in range(self.num_agents)]
        out_a = np.array(out_a).transpose(1, 0, 2)  # (n_threads, n_agents, action_dim)
        return out_a
    
    @torch.no_grad()
    def estimate_value(self, zs, actions, gamma):
        """
        Args:
            zs: [n_agents, (n_threads, num_samples, *dim)]
            actions: [n_agents, (horizon, n_threads, num_samples, action_dim)]
            gamma: scalar, discount factor
        Returns:
            g_returns: [n_agents, (n_threads, num_samples, 1)]
        """
        assert len(zs[0].shape) == 3
        horizon, n_threads, num_samples = actions[0].shape[0], actions[0].shape[1], actions[0].shape[2]
        
        cur_zs = torch.stack([z.clone() for z in zs], dim=-2)  # (n_threads, num_samples, n_agents, dim)
        actions = torch.stack(actions, dim=-2) # (horizon, n_threads, num_samples, n_agents, dim)

        # _returns.shape: [n_agents, (horizon+1, n_threads, num_samples, 1)]
        _returns = [
            torch.zeros(
                horizon + 1,
                n_threads,
                num_samples,
                1,
            ).to(**self.tpdv) for agent_id in range(self.num_agents)
        ]

        for t in range(horizon):
            
            zs_pred = self.dynamics_model.predict(
                z=cur_zs.reshape(n_threads * num_samples, self.num_agents, -1),
                a=actions[t].reshape(n_threads * num_samples, self.num_agents, -1)
            ).reshape(*cur_zs.shape)
            rew_pred = self.reward_model.predict(
                z=cur_zs.reshape(n_threads * num_samples, self.num_agents, -1),
                a=actions[t].reshape(n_threads * num_samples, self.num_agents, -1)
            ).reshape(n_threads, num_samples, -1)
            r_value = self.reward_processor.logits_decode_scalar(rew_pred)
            
            for agent_id in range(self.num_agents):
                _returns[agent_id][t + 1] = _returns[agent_id][t] + gamma ** t * r_value
            
            cur_zs = zs_pred.clone()
            
            

        joint_zs = cur_zs.reshape(n_threads, num_samples, -1)  # (n_threads, num_samples, n_agents*dim)
        joint_actions = torch.cat(
            [
                self.actor[agent_id].get_actions(
                    obs=cur_zs[:, :, agent_id],
                    available_actions=None,
                    stochastic=True
                ) for agent_id in range(self.num_agents)
            ],
            dim=-1,
        )
        horizon_q = self.critic.get_values(
            joint_zs,
            joint_actions,
	        mode="mean"
        )
        for agent_id in range(self.num_agents):
            _returns[agent_id][-1] = _returns[agent_id][-2] + gamma ** horizon * horizon_q

        g_returns = [
            _returns[agent_id][-1].nan_to_num(0)  # (n_threads, num_samples, 1)
            for agent_id in range(self.num_agents)
        ]

        return g_returns

    # TODO
    @torch.no_grad()
    def test_error(self, obs, actions, rewards, next_obs):
        # obs: (n_threads, n_agents, dim)
        joint_act = torch.cat(
            [check(actions[:, agent_id]).to(**self.tpdv) for agent_id in range(self.num_agents)],
            dim=-1
        )
        mean_errors = [[], []]
        for agent_id in range(self.num_agents):
            _obs = check(obs[:, agent_id]).to(**self.tpdv)
            _z = self.obs_encoder[agent_id].encode(_obs)

            _r = check(rewards[:, agent_id]).to(**self.tpdv)

            pred_z = self.dynamics_model[agent_id].predict(
                z=_z,
                a=joint_act
            )
            encode_z = self.obs_encoder[agent_id].encode(check(next_obs[:, agent_id]).to(**self.tpdv))

            pred_r = self.reward_model[agent_id].predict(
                z=_z,
                a=joint_act
            )
            pred_r = self.reward_processor.logits_decode_scalar(pred_r)

            mean_errors[0].append(float((pred_z - encode_z).abs().mean()))
            mean_errors[1].append(float((pred_r - _r).abs().mean()))
        return np.mean(mean_errors[0]), np.mean(mean_errors[1])

    def train(self, train_actor=True):
        """Train the model"""
        if self.buffer.cur_size < self.buffer.batch_size:
            return dict()
        self.total_it += 1
        
        t0 = torch.randperm(self.buffer.cur_size).numpy()[:self.buffer.batch_size]
        self.buffer.update_end_flag()
        indices = [t0]
        for _ in range(self.horizon - 1):
            indices.append(self.buffer.next(indices[-1]))
        
        data_horizon = self.buffer.sample_horizon(horizon=self.horizon, t0=t0)
        (_, sp_obs, sp_actions, _, sp_reward, _, _,
         sp_term, _, sp_next_obs, _, sp_gamma) = data_horizon  # h-step for world model
        
        sp_nstep_reward = np.zeros_like(sp_reward)
        sp_nstep_term = np.zeros_like(sp_term)
        sp_nstep_next_obs = np.zeros_like(sp_next_obs)
        sp_nstep_gamma = np.zeros_like(sp_gamma)
        for t, indice in enumerate(indices):
            data_nstep = self.buffer.sample(indice=indice)
            (_, _, _, _, nstep_reward, _, _,
             nstep_term, _, nstep_next_obs, _, nstep_gamma, _, _) = data_nstep  # compute q-target
            
            sp_nstep_reward[t] = nstep_reward  # (batch_size, 1)
            sp_nstep_term[t] = nstep_term  # (batch_size, 1)
            sp_nstep_next_obs[:, t] = nstep_next_obs  # (n_agents, batch_size, dim) -> (n_agents, horizon, batch_size, dim)
            sp_nstep_gamma[t] = nstep_gamma  # (batch_size, 1)
        
        train_info, zs = self.model_train(
            sp_obs, sp_actions, sp_reward, sp_next_obs,
            sp_nstep_reward, sp_nstep_term, sp_nstep_next_obs, sp_nstep_gamma,
        )

        if self.total_it % self.policy_freq == 0 and train_actor:
            _info = self.actor_train(
                zs=[zs[_i, :-1] for _i in range(self.num_agents)]
            )
            train_info.update(_info)

        self.critic.soft_update()

        return train_info

    def model_train(self, obs, actions, reward, next_obs,
                    nstep_reward, nstep_term, nstep_next_obs, nstep_gamma,):
        """Update the world-model and Q function"""
        
        # region (n_agents, horizon, batch_size, *dim)
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        # joint_actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)  # (horizon, batch_size, dim * n_agents)
        reward = check(reward).to(**self.tpdv)  # (horizon, batch_size, 1)
        next_obs = check(next_obs).to(**self.tpdv)  # (n_agents, horizon, batch_size, dim)
        batch_size = obs.shape[2]

        nstep_reward = check(nstep_reward).to(**self.tpdv)  # (batch_size, 1)
        nstep_term = check(nstep_term).to(**self.tpdv)  # (batch_size, 1)
        nstep_next_obs = check(nstep_next_obs).to(**self.tpdv)  # (n_agents, batch_size, dim)
        nstep_gamma = check(nstep_gamma).to(**self.tpdv)  # (batch_size, 1)
        
        n_agents, horizon, batch_size, _ = obs.shape
        # endregion

        # region get target
        with torch.no_grad():
            next_zs = [
                self.obs_encoder[_i].encode(next_obs[_i])
                for _i in range(self.num_agents)
            ]
            next_nstep_zs = [
                self.obs_encoder[_i].encode(nstep_next_obs[_i])
                for _i in range(self.num_agents)
            ]
            next_nstep_actions = [
                self.actor[_i].get_actions(next_zs[_i])
                for _i in range(self.num_agents)
            ]
            
            next_zs = torch.stack(next_zs, dim=0)  # (n_agents, horizon, batch_size, dim)
        
        next_q_values = self.critic.get_target_values(
            share_obs=torch.cat(next_nstep_zs, dim=-1),
            actions=torch.cat(next_nstep_actions, dim=-1)
        )
        q_targets = nstep_reward + nstep_gamma * next_q_values * (1 - nstep_term)  # (horizon, batch_size, 1)
        self.model_turn_on_grad()
        # endregion
        
        train_info = {}

        # region computes h-step prediction loss
        dynamics_loss, reward_loss, q_loss, balance_loss = 0.0, 0.0, 0.0, 0.0
        train_info = {"reward_acc": 0.0, "reward_err": 0.0}
        
        zs = torch.zeros(self.num_agents, self.horizon + 1, batch_size, self.latent_dim).to(**self.tpdv)
        for agent_id in range(self.num_agents):
            zs[agent_id, 0] = self.obs_encoder[agent_id].encode(obs[agent_id, 0])
        
        for t in range(self.horizon):
            z_pred = self.dynamics_model.predict(
                zs[:, t].permute(1, 0, 2),  # (na, h, b, z) -> (b, na, z)
                actions[:, t].permute(1, 0, 2)  # (na, h, b, a) -> (b, na, a)
            ).permute(1, 0, 2)  # (b, na, z) -> (na, b, z)
            reward_pred_logits, aux = self.reward_model.forward(
                zs[:, t].permute(1, 0, 2),
                actions[:, t].permute(1, 0, 2)
            )  # (b, num_bins)
            
            dynamics_loss += F.mse_loss(
                z_pred, next_zs[:, t]
            ).mean() * (self.step_rho ** t)
            reward_loss += self.reward_processor.dis_reg_loss(
                logits=reward_pred_logits,
                target=reward[t]
            ).mean() * (self.step_rho ** t)
            balance_loss += aux["loss_balancing"] * (self.step_rho ** t)
            
            
            zs[:, t + 1] = z_pred.clone()
            
            with torch.no_grad():
                _error = torch.abs(reward[t] - self.reward_processor.logits_decode_scalar(reward_pred_logits))
                train_info["reward_acc"] += (_error <= 0.05).sum().item() / _error.shape[0] / self.horizon
                train_info["reward_err"] += torch.mean(_error).item() / self.num_agents / self.horizon
            
            joint_z_pred = torch.cat([zs[_i, t] for _i in range(self.num_agents)], dim=-1)
            joint_action_t = torch.cat([actions[_i, t] for _i in range(self.num_agents)], dim=-1)
            q_pred_logits = self.critic.critic(joint_z_pred, joint_action_t)
            q_pred_logits2 = self.critic.critic2(joint_z_pred, joint_action_t)
            q_loss += ((self.critic.processor.dis_reg_loss(logits=q_pred_logits, target=q_targets[t]).mean() +
                        self.critic.processor.dis_reg_loss(logits=q_pred_logits2, target=q_targets[t]).mean()) / 2) * (
                                  self.step_rho ** t)
        
        dynamics_loss /= self.horizon
        reward_loss /= self.horizon
        q_loss /= self.horizon
        balance_loss /= self.horizon
        # endregion
        
        train_info["dynamics_loss"] = float(dynamics_loss)
        train_info["reward_loss"] = float(reward_loss)
        train_info["q_loss"] = float(q_loss)
        train_info["balance_loss"] = float(balance_loss)
        
        total_loss = (
                q_loss * self.algo_args["world_model"]["q_coef"]
                + reward_loss * self.algo_args["world_model"]["reward_coef"]
                + dynamics_loss * self.algo_args["world_model"]["dynamics_coef"]
                + balance_loss * self.algo_args["world_model"]["balance_coef"]
        )
        train_info["total_loss"] = (
                train_info["q_loss"] * self.algo_args["world_model"]["q_coef"]
                + train_info["reward_loss"] * self.algo_args["world_model"]["reward_coef"]
                + train_info["dynamics_loss"] * self.algo_args["world_model"]["dynamics_coef"]
                + train_info["balance_loss"] * self.algo_args["world_model"]["balance_coef"]
        )
        
        self.model_optimizer.zero_grad()
        total_loss.backward()
        for group in self.model_optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 20)
        self.model_optimizer.step()
        
        self.model_turn_off_grad()
        
        return train_info, zs.detach()

    def actor_train(self, zs):
        """Train the actor network"""
        train_info = {
            "actor_loss": [0.0] * self.num_agents,
        }

        # fill actions via before-trained actor
        actions = []
        logp_actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                action, logp_action = self.actor[agent_id].get_actions_with_logprobs(
                    obs=zs[agent_id],
                    available_actions=None,
                )
                actions.append(action)
                logp_actions.append(logp_action)
        # actions shape: (n_agents, batch_size, dim), logp_actions shape: (n_agents, batch_size, 1)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(np.random.permutation(self.num_agents))

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()

            # get action(with grad) via before-trained actor
            actions[agent_id], logp_actions[agent_id] = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                zs[agent_id],
                None,
            )
            logp_action = logp_actions[agent_id]
            value_pred = self.critic.get_values(
                torch.cat(zs, dim=-1), torch.cat(actions, dim=-1), "mean"
            )  # (horizon, batch_size, 1)
            
            self.critic.scale.update(value_pred[0])
            value_pred = self.critic.scale(value_pred)

            actor_loss = torch.zeros(1).to(**self.tpdv)
            for t in range(len(zs[agent_id])):
                actor_loss += (self.entropy_coef * logp_action[t] - value_pred[t]).mean() * (self.step_rho ** t)
            actor_loss /= len(zs[agent_id])

            self.actor[agent_id].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[agent_id].actor_optimizer.step()

            self.actor[agent_id].turn_off_grad()

            # update action via after-trained actor
            actions[agent_id], _ = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                zs[agent_id],
                None,
            )

            train_info["actor_loss"][agent_id] = actor_loss.item()
            train_info["pi_scale"] = self.critic.scale.value

        return train_info

    def init_config(self):
        task = get_task_name(self.args["env"], self.env_args)
        hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        run_dir = str(os.path.join(
		    self.algo_args["logger"]["log_dir"],
		    self.args["env"],
		    task,
		    self.args["algo"],
		    "-".join([hms_time, "seed-{:0>3}".format(self.algo_args["seed"]["seed"])]),
        ))

        os.makedirs(run_dir, exist_ok=True)
        save_config(self.args, self.algo_args, self.env_args, run_dir)

        save_dir = os.path.join(run_dir, "models")
        if self.algo_args["logger"]["save_model"]:
            os.makedirs(save_dir, exist_ok=True)

        log_file = open(os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8")
        
        expt_name = "-".join([hms_time, "seed-{:0>3}".format(self.algo_args["seed"]["seed"])])
        return run_dir, save_dir, log_file, task, expt_name

    def console_log(self, _step, **kwargs):
        if self.algo_args["logger"]["wandb"]:
            for key, value in kwargs.items():
                if value == value:  # nan
                    wandb.log(value, step=self.total_it)

        print_lines = [
            "",
            "******** iter: %s, steps: %s/%s, iter_time: %.1fs, total_time: " % (
                convert_num(self.total_it),
                convert_num(_step * self.algo_args["train"]["n_rollout_threads"]),
                convert_num(self.algo_args["train"]["num_env_steps"]),
                time.time() - self._check_time,
            ) + str(datetime.timedelta(seconds=int(time.time() - self._start_time))) + " ********"
        ]

        for key, value in kwargs.items():
            print_lines.append("%s" % key + "".join([", %s: %.6f" % (k, v) for k, v in value.items()]))

        for line in print_lines:
            print(line)
            self.log_file.write(line + "\n")
        self.log_file.flush()

        self._check_time = time.time()

    @torch.no_grad()
    def eval(self, step, use_plan=False):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int32
        )

        mean_err = [[], []]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        t0 = [True] * self.algo_args["eval"]["n_eval_rollout_threads"]

        while True:
            if use_plan:
                eval_actions = self.plan(
                    eval_obs,
                    t0=t0,
                    available_actions=eval_available_actions,
                    add_random=False
                )
            else:
                eval_actions = self.get_actions(
                    eval_obs,
                    available_actions=eval_available_actions,
                    add_random=False
                )

            (
                eval_new_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            # print(eval_rewards)
            # if use_plan:
            #     _d_err, _r_err = self.test_error(eval_obs, eval_actions, eval_rewards, eval_obs)
            #     mean_err[0].append(_d_err)
            #     mean_err[1].append(_r_err)

            eval_obs = eval_new_obs.copy()

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

                    t0[eval_i] = True
                else:
                    t0[eval_i] = False

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                eval_info = {
                    "eval_avg_rew": eval_avg_rew,
                    "eval_avg_len": eval_avg_len,
                }
                if use_plan:
                    eval_info["mean_dynamics_error"] = np.mean(mean_err[0])
                    eval_info["mean_reward_error"] = np.mean(mean_err[1])

                self.console_log(
                    step,
                    eval_info=eval_info
                )
                break

    def restore(self):
        """Restore the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(
                self.algo_args["train"]["model_dir"],
                agent_id
            )
        self.critic.restore(self.algo_args["train"]["model_dir"])

        for agent_id in range(self.num_agents):
            self.obs_encoder[agent_id].restore(
                load_dir=self.algo_args["train"]["model_dir"],
                agent_id=agent_id,
                model_name="obs_encoder"
            )
        self.dynamics_model.restore(
            load_dir=self.algo_args["train"]["model_dir"],
            model_name="dynamics_model"
        )
        self.reward_model.restore(
            load_dir=self.algo_args["train"]["model_dir"],
            model_name="reward_model"
        )

    def save(self):
        """Save the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)

        for agent_id in range(self.num_agents):
            self.obs_encoder[agent_id].save(save_dir=self.save_dir, agent_id=agent_id, model_name="obs_encoder")
        self.dynamics_model.save(save_dir=self.save_dir, model_name="dynamics_model")
        self.reward_model.save(save_dir=self.save_dir, model_name="reward_model")

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.log_file.close()

        if self.algo_args["logger"].get("wandb", False):
            wandb.finish()
            print("\nwandb run has finished")
    
    def model_turn_on_grad(self):
        for agent_id in range(self.num_agents):
            self.obs_encoder[agent_id].turn_on_grad()
        self.dynamics_model.turn_on_grad()
        self.reward_model.turn_on_grad()
        self.critic.turn_on_grad()

    def model_turn_off_grad(self):
        for agent_id in range(self.num_agents):
            self.obs_encoder[agent_id].turn_off_grad()
        self.dynamics_model.turn_off_grad()
        self.reward_model.turn_off_grad()
        self.critic.turn_off_grad()
