from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry


from legged_gym.envs.g1.g1_config import g1Cfg, g1CfgPPO
from legged_gym.envs.g1.g1_env import g1Env

from legged_gym.envs.go2.go2_config import go2Cfg, go2CfgPPO
from legged_gym.envs.go2.go2_env import go2Env

from legged_gym.envs.x2.x2_config import x2Cfg, x2CfgPPO
from legged_gym.envs.x2.x2_env import x2Env


task_registry.register( "g1", g1Env, g1Cfg(), g1CfgPPO() )
task_registry.register( "go2", go2Env, go2Cfg(), go2CfgPPO() )
task_registry.register( "x2", x2Env, x2Cfg(), x2CfgPPO() )
