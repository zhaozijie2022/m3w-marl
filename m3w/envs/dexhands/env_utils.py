from isaacgym import gymapi
from isaacgym import gymutil
import os
import argparse
import yaml
import utils.pytorch_utils as ptu
from m3w.envs.dexhands.bidexhands.tasks.hand_base.multi_vec_task import MultiVecTaskPython


from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_over import ShadowHandOver
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarm
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_catch_abreast import ShadowHandCatchAbreast
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_door_close_inward import ShadowHandDoorCloseInward
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_door_open_inward import ShadowHandDoorOpenInward
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_scissors import ShadowHandScissors
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_pen import ShadowHandPen
from m3w.envs.dexhands.bidexhands.tasks.shadow_hand_kettle import ShadowHandKettle


def make_single_env(env_args, task_name):
    # env_args provide seed, n_envs_per_task

    # region get_args() <- bidexhands/utils/config.py
    parser = argparse.ArgumentParser()

    parser.add_argument("--expt", type=str, default="default")
    parser.add_argument("--env", type=str, default="mt_dexhands")
    parser.add_argument("--algo", type=str, default="ma_world_model")

    parser.add_argument("--test", action="store_true", default=False, help="Run trained policy, no training")
    parser.add_argument("--play", action="store_true", default=False,
                        help="Run trained policy, the same as test, can be used only by rl_games RL library")
    parser.add_argument("--resume", type=int, default=0, help="Resume training or start testing from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="Base", help="Path to the saved weights, only for rl_games RL library")
    parser.add_argument("--headless", action="store_true", default=True, help="Force display off at all times")
    parser.add_argument("--horovod", action="store_true", default=False,
                        help="Use horovod for multi-gpu training, have effect only with rl_games RL library")
    parser.add_argument("--task", type=str, default="Humanoid",
                        help="Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity")
    parser.add_argument("--task_type", type=str, default="MultiAgent", help="Choose Python or C++")
    parser.add_argument("--rl_device", type=str, default="cuda:0", help="Choose CPU or GPU device for inferencing policy network")
    parser.add_argument("--logdir", type=str, default="logs/")
    parser.add_argument("--experiment", type=str, default="Base",
                        help="Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name")
    parser.add_argument("--metadata", action="store_true", default=False,
                        help="Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user")
    parser.add_argument("--cfg_train", type=str, default="Base")
    parser.add_argument("--cfg_env", type=str, default="Base")
    parser.add_argument("--num_envs", type=int, default=0, help="Number of environments to create - override config file")
    parser.add_argument("--episode_length", type=int, default=0, help="Episode length, by default is read from yaml config")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max_iterations", type=int, default=-1, help="Set a maximum number of training iterations")
    parser.add_argument("--steps_num", type=int, default=-1,
                        help="Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings.")
    parser.add_argument("--minibatch_size", type=int, default=-1,
                        help="Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings.")
    parser.add_argument("--randomize", action="store_true", default=False, help="Apply physics domain randomization")
    parser.add_argument("--torch_deterministic", action="store_true", default=False,
                        help="Apply additional PyTorch settings for more deterministic behaviour")
    parser.add_argument("--model_dir", type=str, default="", help="Choose a model dir")
    parser.add_argument("--datatype", type=str, default="random", help="Choose an offline datatype")
    # custom_parameters = [
    #     {"name": "--expt", "type": str, "default": "default",},
    #     {"name": "--env", "type": str, "default": "mt_dexhands",},
    #     {"name": "--algo", "type": str, "default": "ma_world_model",},
    #
    #     {"name": "--test", "action": "store_true", "default": False,
    #      "help": "Run trained policy, no training"},
    #     {"name": "--play", "action": "store_true", "default": False,
    #      "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
    #     {"name": "--resume", "type": int, "default": 0,
    #      "help": "Resume training or start testing from a checkpoint"},
    #     {"name": "--checkpoint", "type": str, "default": "Base",
    #      "help": "Path to the saved weights, only for rl_games RL library"},
    #     {"name": "--headless", "action": "store_true", "default": True,
    #      "help": "Force display off at all times"},
    #     {"name": "--horovod", "action": "store_true", "default": False,
    #      "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
    #     {"name": "--task", "type": str, "default": "Humanoid",
    #      "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
    #     {"name": "--task_type", "type": str,
    #      "default": "MultiAgent", "help": "Choose Python or C++"},
    #     {"name": "--rl_device", "type": str, "default": "cuda:0",
    #      "help": "Choose CPU or GPU device for inferencing policy network"},
    #     {"name": "--logdir", "type": str, "default": "logs/"},
    #     {"name": "--experiment", "type": str, "default": "Base",
    #      "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
    #     {"name": "--metadata", "action": "store_true", "default": False,
    #      "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
    #     {"name": "--cfg_train", "type": str,
    #      "default": "Base"},
    #     {"name": "--cfg_env", "type": str, "default": "Base"},
    #     {"name": "--num_envs", "type": int, "default": 0,
    #      "help": "Number of environments to create - override config file"},
    #     {"name": "--episode_length", "type": int, "default": 0,
    #      "help": "Episode length, by default is read from yaml config"},
    #     {"name": "--seed", "type": int, "help": "Random seed"},
    #     {"name": "--max_iterations", "type": int, "default": -1,
    #      "help": "Set a maximum number of training iterations"},
    #     {"name": "--steps_num", "type": int, "default": -1,
    #      "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
    #     {"name": "--minibatch_size", "type": int, "default": -1,
    #      "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
    #     {"name": "--randomize", "action": "store_true", "default": False,
    #      "help": "Apply physics domain randomization"},
    #     {"name": "--torch_deterministic", "action": "store_true", "default": False,
    #      "help": "Apply additional PyTorch settings for more deterministic behaviour"},
    #     {"name": "--model_dir", "type": str, "default": "",
    #      "help": "Choose a model dir"},
    #     {"name": "--datatype", "type": str, "default": "random",
    #      "help": "Choose an offline datatype"}
    # ]
    # args = gymutil.parse_arguments(
    #     description="RL Policy",
    #     custom_parameters=custom_parameters)
    #
    # parser = argparse.ArgumentParser(description="RL Policy")
    # if args.headless:
    #     parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    # if no_graphics:
    parser.add_argument('--nographics', action='store_true', default=True,
                        help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    args = parser.parse_args([])

    args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # # Using --nographics implies --headless
    # if no_graphics and args.nographics:
    #     args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    args.task = task_name
    args.algo = 'm3w'  # placeholder
    args.rl_device = ptu.device
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
    args.train = True

    assert "MetaMT" not in args.task, "MetaMT is not supported"
    args.cfg_env = "envs/mt_dexhands/bidexhands/cfg/{}.yaml".format(args.task)
    # endregion

    # region load_cfg() <- bidexhands/utils/config.py\

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if env_args["n_envs_per_task"] > 0:
        cfg["env"]["numEnvs"] = env_args["n_envs_per_task"]

    cfg["env"]["asset"]["assetRoot"] = "envs/mt_dexhands/assets"

    cfg["name"] = args.task
    cfg["headless"] = args.headless
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}
    # endregion

    # region parse_sim_params() <- bidexhands/utils/config.py
    sim_params = gymapi.SimParams()
    sim_params.dt = 1. / 60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    # endregion

    # region get_AgentIndex() <- bidexhands/utils/process_marl.py
    agent_index = [eval(cfg["env"]["handAgentIndex"]), eval(cfg["env"]["handAgentIndex"])]
    # endregion

    # region parse_task() <- bidexhands/utils/parse_task.py
    cfg["seed"] = env_args["seed"]
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
        agent_index=agent_index,
        is_multi_agent=True)

    env = MultiVecTaskPython(task, args.rl_device)
    # endregion

    return env




