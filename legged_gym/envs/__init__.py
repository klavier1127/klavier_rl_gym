from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry


from .g1.g1_config import g1Cfg, g1CfgPPO
from .g1.g1_env import g1Env

from .go2.go2_config import go2Cfg, go2CfgPPO
from .go2.go2_env import go2Env

from .go2_amp.go2_amp_config import go2AMPCfg, go2AMPCfgPPO
from .go2_amp.go2_amp_env import go2AMPEnv

from .x2.x2_config import x2Cfg, x2CfgPPO
from .x2.x2_env import x2Env

from .tita.tita_config import titaCfg, titaCfgPPO
from .tita.tita_env import titaEnv


task_registry.register( "g1", g1Env, g1Cfg(), g1CfgPPO() )
task_registry.register( "go2", go2Env, go2Cfg(), go2CfgPPO() )
task_registry.register( "go2_amp", go2AMPEnv, go2AMPCfg(), go2AMPCfgPPO() )
task_registry.register( "x2", x2Env, x2Cfg(), x2CfgPPO() )
task_registry.register( "tita", titaEnv, titaCfg(), titaCfgPPO() )

