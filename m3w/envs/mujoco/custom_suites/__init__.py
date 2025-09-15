from .cheetah import HalfCheetahMulti
from .humanoid import HumanoidMulti
from .humanoid_standup import HumanoidStandupMulti
from .reacher import ReacherMulti
from .swimmer import SwimmerMulti
from .walker2d import Walker2dMulti

ENV_REGISTRY = {
    "2_cheetah": HalfCheetahMulti,
    "2_humanoid": HumanoidMulti,
    "2_humanoid_standup": HumanoidStandupMulti,
    "2_reacher": ReacherMulti,
    "2_swimmer": SwimmerMulti,
    "2_walker2d": Walker2dMulti
}

ARGS_REGISTRY = {
    "2_cheetah": {"scenario": "HalfCheetah-v2", "agent_conf": "2x3", "agent_obsk": 0},
    "2_humanoid": {"scenario": "Humanoid-v2", "agent_conf": "9|8", "agent_obsk": 0},
    "2_humanoid_standup": {"scenario": "HumanoidStandup-v2", "agent_conf": "9|8", "agent_obsk": 0},
    "2_reacher": {"scenario": "Reacher-v2", "agent_conf": "2x1", "agent_obsk": 0},
    "2_swimmer": {"scenario": "Swimmer-v2", "agent_conf": "2x1", "agent_obsk": 0},
    "2_walker2d": {"scenario": "Walker2d-v2", "agent_conf": "2x3", "agent_obsk": 0}
}
