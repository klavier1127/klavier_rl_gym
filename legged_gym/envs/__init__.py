# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.h1.h1_config import h1Cfg, h1CfgPPO
from legged_gym.envs.h1.h1_env import h1Env

from legged_gym.envs.g1.g1_config import g1Cfg, g1CfgPPO
from legged_gym.envs.g1.g1_env import g1Env

from legged_gym.envs.g1_amp.g1_amp_config import g1AMPCfg, g1AMPCfgPPO
from legged_gym.envs.g1_amp.g1_amp_env import g1AMPEnv

from legged_gym.envs.go2.go2_config import go2Cfg, go2CfgPPO
from legged_gym.envs.go2.go2_env import go2Env

from legged_gym.envs.go2_amp.go2_amp_config import go2AMPCfg, go2AMPCfgPPO
from legged_gym.envs.go2_amp.go2_amp_env import go2AMPEnv

from legged_gym.envs.go2w.go2w_config import go2wCfg, go2wCfgPPO
from legged_gym.envs.go2w.go2w_env import go2wEnv

from legged_gym.envs.xbot.xbot_config import xbotCfg, xbotCfgPPO
from legged_gym.envs.xbot.xbot_env import xbotEnv

from legged_gym.envs.bx.bx_config import bxCfg, bxCfgPPO
from legged_gym.envs.bx.bx_env import bxEnv

task_registry.register( "h1", h1Env, h1Cfg(), h1CfgPPO() )
task_registry.register( "g1", g1Env, g1Cfg(), g1CfgPPO() )
task_registry.register( "g1_amp", g1AMPEnv, g1AMPCfg(), g1AMPCfgPPO() )
task_registry.register( "go2", go2Env, go2Cfg(), go2CfgPPO() )
task_registry.register( "go2_amp", go2AMPEnv, go2AMPCfg(), go2AMPCfgPPO() )
task_registry.register( "go2w", go2wEnv, go2wCfg(), go2wCfgPPO() )
task_registry.register( "bx", bxEnv, bxCfg(), bxCfgPPO() )
task_registry.register( "xbot", xbotEnv, xbotCfg(), xbotCfgPPO() )