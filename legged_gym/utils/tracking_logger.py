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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

# 计算图像的尺寸和 DPI
width_in_inches = 2048 / 100  # 宽度（英寸）
height_in_inches = 1080 / 100  # 高度（英寸）
dpi = 100  # 每英寸像素点数

class TrackingLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def reset(self):
        self.state_log.clear()


    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 5
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(width_in_inches * 5, height_in_inches * 2))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and measured positions
        a = axs[0, 0]
        if log["hip_r"]: a.plot(time, log["hip_r"], label='measured')
        if log["hip_r_target"]: a.plot(time, log["hip_r_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='right hip pitch tracking')
        a.legend()
        # plot joint velocity
        a = axs[0, 1]
        if log["knee_r"]: a.plot(time, log["knee_r"], label='measured')
        if log["knee_r_target"]: a.plot(time, log["knee_r_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='right knee pitch tracking')
        a.legend()
        # plot base vel z
        a = axs[0, 2]
        if log["ankle_r"]: a.plot(time, log["ankle_r"], label='measured')
        if log["ankle_r_target"]: a.plot(time, log["ankle_r_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='right ankle pitch tracking')
        a.legend()
        # plot base vel z
        a = axs[0, 3]
        if log["hipyaw_r"]: a.plot(time, log["hipyaw_r"], label='measured')
        if log["hipyaw_r_target"]: a.plot(time, log["hipyaw_r_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='right hip yaw tracking')
        a.legend()
        # plot base vel z
        a = axs[0, 4]
        if log["hiproll_r"]: a.plot(time, log["hiproll_r"], label='measured')
        if log["hiproll_r_target"]: a.plot(time, log["hiproll_r_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='right hip roll tracking')
        a.legend()
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["hip_l"]: a.plot(time, log["hip_l"], label='measured')
        if log["hip_l_target"]: a.plot(time, log["hip_l_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='left hip pitch tracking')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["knee_l"]: a.plot(time, log["knee_l"], label='measured')
        if log["knee_l_target"]: a.plot(time, log["knee_l_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='left knee pitch tracking')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["ankle_l"]: a.plot(time, log["ankle_l"], label='measured')
        if log["ankle_l_target"]: a.plot(time, log["ankle_l_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='left ankle pitch tracking')
        a.legend()

        # plot base vel z
        a = axs[1, 3]
        if log["hipyaw_l"]: a.plot(time, log["hipyaw_l"], label='measured')
        if log["hipyaw_l_target"]: a.plot(time, log["hipyaw_l_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='left hip yaw tracking')
        a.legend()

        # plot base vel z
        a = axs[1, 4]
        if log["hiproll_l"]: a.plot(time, log["hiproll_l"], label='measured')
        if log["hiproll_l_target"]: a.plot(time, log["hiproll_l_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='left hip roll tracking')
        a.legend()

        # plot eu_ang , ang_vx, x
        a = axs[2, 4]
        if log["eu_x"]: a.plot(time, log["eu_x"], label='eu_anglex')
        if log["ang_vx"]: a.plot(time, log["ang_vx"], label='ang_vx')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='euler angle and ang_v roll tracking')
        a.legend()

        a = axs[2, 0]
        if log["eu_y"]: a.plot(time, log["eu_y"], label='eu_angley')
        if log["ang_vy"]: a.plot(time, log["ang_vy"], label='ang_vy')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='euler angle and ang_v pitch tracking')
        a.legend()

        #plot sim and real tracking
        a = axs[2, 1]
        if log["sim_baseline"]: a.plot(time, log["sim_baseline"], label='sim')
        if log["real_baseline"]: a.plot(time, log["real_baseline"], label='real')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='sim real roll tracking')
        a.legend()




        plt.show()



    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()