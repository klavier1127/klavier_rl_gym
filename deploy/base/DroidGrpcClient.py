import os
import sys
import math
import time
import asyncio
import threading
from base64 import b64encode
# from foxglove_websocket.server import FoxgloveServer
from grpc import insecure_channel

sys.path.append(os.path.abspath("/home/droid/x02gym/projects/deploy"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../protos'))

from deploy.protos import ecatplat_service_pb2_grpc as service_pb2_grpc
from deploy.protos import motion_control_pb2 as msg_pb2

with open(os.path.join(os.path.dirname(msg_pb2.__file__), "motion_control.bin"), "rb") as schema_bin:
    schema_base64 = b64encode(schema_bin.read()).decode("ascii")


class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break


class DroidGrpcClient:
    def __init__(self, _channel):
        self.fox_running = True
        self.thread = threading.Thread(target=self.sync_proto)

        self.num_actions = 18
        self.stub = service_pb2_grpc.EcatplatServiceStub(_channel)
        self.robotConfigs = msg_pb2.DroidConfigs()
        self.robotState = msg_pb2.DroidStateResponse()
        self.robotCommand = msg_pb2.DroidCommandRequest()

        # 初始化 robotCommand
        self.get_robot_config()
        self.get_robot_state()
        self.robotCommand.cmd_enable = 1
        for idx in range(self.num_actions):
            self.robotCommand.mode.append(1)
            self.robotCommand.position.append(self.robotState.position[idx])
            self.robotCommand.velocity.append(0.0)
            self.robotCommand.torque.append(0.0)
            self.robotCommand.ens.append(1)
            self.robotCommand.kp.append(self.robotConfigs.kp[idx])
            self.robotCommand.kd.append(self.robotConfigs.kd[idx])
            self.robotCommand.max_torque.append(self.robotConfigs.imax[idx])

    def fox_run(self):
        self.fox_running = True
        self.thread.start()

    def fox_stop(self):
        self.fox_running = False
        self.thread.join()

    # async def proto(self):
        # Specify the server's host, port, and a human-readable name
        # async with FoxgloveServer("0.0.0.0", 8765, "example server") as server:
        #     cmd_ch_id = await server.add_channel({
        #             "topic": "x02_cmd",
        #             "encoding": "protobuf",
        #             "schemaName": "droidgrpc.DroidCommandRequest",
        #             "schema": schema_base64,
        #         }
        #     )
        #     state_ch_id = await server.add_channel({
        #             "topic": "x02_state",
        #             "encoding": "protobuf",
        #             "schemaName": "droidgrpc.DroidStateResponse",
        #             "schema": schema_base64,
        #         }
        #     )
        #     while self.fox_running:
        #         await asyncio.sleep(0.01)
        #         await server.send_message(
        #             cmd_ch_id,
        #             time.time_ns(),
        #             self.robotCommand.SerializeToString(),
        #         )
        #         await server.send_message(
        #             state_ch_id,
        #             time.time_ns(),
        #             self.robotState.SerializeToString(),
        #         )

    def sync_proto(self):
        loop = asyncio.new_event_loop()  # 创建新的事件循环
        asyncio.set_event_loop(loop)  # 将新的事件循环设置为当前线程的默认事件循环
        loop.run_until_complete(self.proto())

    def get_robot_config(self):
        empty_request = msg_pb2.Empty()
        self.robotConfigs = self.stub.GetRobotConfig(empty_request)
        # self.print_robot_configs()

    def get_robot_state(self):
        empty_request = msg_pb2.Empty()
        self.robotState = self.stub.GetRobotState(empty_request)
        # self.print_robot_state()

    def get_robot_state_stream(self):
        empty_request = msg_pb2.Empty()
        for response in self.stub.GetRobotStateStream(empty_request):
            self.print_robot_state()

    def set_robot_command(self):
        response = self.stub.SetRobotCommand(self.robotCommand)
        if not response:  # Assuming the RPC method returns a response
            print("RPC failed")

    # 创建一个生成器，生成DroidCommandRequest消息
    def generate_commands(self):
        for _ in range(5):  # 假设我们发送5个命令
            yield self.robotCommand
            print("RPC succeeded")
            time.sleep(1)  # 等待一段时间

    def set_robot_command_stream(self):
        response = self.stub.SetRobotCommandStream(self.generate_commands())
        # 检查响应
        if response:
            print("Command stream sent successfully.")
        else:
            print("Failed to send command stream.")

    def exchange_robot_control_stream(self):
        responses = self.stub.ExchangeRobotControlStream(self.generate_commands())
        for response in responses:
            print("Received state response:")
            print(f"System TIC: {response.system_tic}")
            for pos in response.position:
                print(f"Position: {pos}")

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.num_actions
        for idx in range(10):
            q0[idx] = self.robotState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_robot_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0

            for idx in range(10):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.robotCommand.position[idx] = qt
            self.set_robot_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def print_robot_configs(self):
        line = '-' * 200
        print("---------+" + line)
        print("MtrName  |", end="")
        for i in range(len(self.robotConfigs.joint_name)):
            print(f"{self.robotConfigs.joint_name[i]:>11}", end="")
        print()
        print("---------+" + line)
        for attr in ["ecatid", "channel", "deviceid", "aeofst", "uie", "uic", "uae", "pmin", "pmax", "ae0", "kp", "kd",
                     "imax"]:
            print("{:<8} |".format(attr.upper()), end="")
            for i in range(self.num_actions):
                print("{:>11.3f}".format(getattr(self.robotConfigs, attr)[i]), end="")
            print()

        print("---------+" + line)

    def print_robot_state(self):
        line = '-' * 200
        print(f"system tic: {self.robotState.system_tic} ms")
        print("---------+" + line)
        print("MtrName  |", end="")
        for i in range(len(self.robotConfigs.joint_name)):
            print(f"{self.robotConfigs.joint_name[i]:>11}", end="")
        print()
        print("---------+" + line)

        labels = ["qc", "dqc", "tqc", "temp", "absc", "loss"]
        for label in labels:
            print(f"{label:<8} |", end="")
            for i in range(len(self.robotConfigs.joint_name)):
                if label == "qc":
                    print(f"{self.robotState.position[i]:>11.3f}", end="")
                elif label == "dqc":
                    print(f"{self.robotState.velocity[i]:>11.3f}", end="")
                elif label == "tqc":
                    print(f"{self.robotState.torque[i]:>11.3f}", end="")
                elif label == "temp":
                    print(f"{self.robotState.temperature[i]:>11.3f}", end="")
                elif label == "absc":
                    print(f"{self.robotState.abs_encoder[i]:>11.3f}", end="")
                elif label == "loss":
                    print(f"{self.robotState.pack_loss[i]:>11}", end="")
            print()
        print("---------+" + line)
        line = '-' * 80
        print(
            f"Foot Sensor (L L R R):      {self.robotState.foot_force[0]:>10.3f} {self.robotState.foot_force[1]:>10.3f} {self.robotState.foot_force[2]:>15.3f} {self.robotState.foot_force[3]:>10.3f}")
        print(line)

        print(f"Imu pack stamp: {self.robotState.imu_stamp:<10}")
        print(
            f"Accelerometer (m/s^2): {self.robotState.imu_acc[0]:>17.3f} {self.robotState.imu_acc[1]:>17.3f} {self.robotState.imu_acc[2]:>17.3f}")
        print(
            f"Attitude      (Euler): {self.robotState.imu_euler[0]:>17.3f} {self.robotState.imu_euler[1]:>17.3f} {self.robotState.imu_euler[2]:>17.3f}")
        print(
            f"Gyroscope     (rad/s): {self.robotState.imu_gyro[0]:>17.3f} {self.robotState.imu_gyro[1]:>17.3f} {self.robotState.imu_gyro[2]:>17.3f}")
        print(line)

        print(
            f"Attitude(est) (Euler): {self.robotState.est_euler[0]:>17.3f} {self.robotState.est_euler[1]:>17.3f} {self.robotState.est_euler[2]:>17.3f}")
        print(
            f"COM Pos(est)      (m): {self.robotState.est_com_pos[0]:>17.3f} {self.robotState.est_com_pos[1]:>17.3f} {self.robotState.est_com_pos[2]:>17.3f}")
        print(
            f"COM Vel(est)    (m/s): {self.robotState.est_com_vel[0]:>17.3f} {self.robotState.est_com_vel[1]:>17.3f} {self.robotState.est_com_vel[2]:>17.3f}")
        print(line)

        print(
            f"Bus Information:      {self.robotState.bus_voltage:>18.3f} {self.robotState.bus_current:>17.3f} {self.robotState.bus_energy:>17.3f}")
        print(line)

        print(
            f"Remote Controller: {self.robotState.rc_du[0]:>21} {self.robotState.rc_du[1]:>10} {self.robotState.rc_du[2]:>13} {self.robotState.rc_du[3]:>10}")
        print(
            f"{self.robotState.rc_keys[0]:>40} {self.robotState.rc_keys[1]:>10} {self.robotState.rc_keys[2]:>13} {self.robotState.rc_keys[3]:>10}")
        print(line)


if __name__ == '__main__':
    # channel = insecure_channel('localhost:50051')
    channel = insecure_channel('192.168.55.110:50051')
    gBot = DroidGrpcClient(channel)
    gBot.fox_run()
    time.sleep(2)
    gBot.get_robot_config()
    gBot.get_robot_state()
    # gBot.set_robot_command()
    # gBot.set_robot_command_stream()
    # gBot.exchange_robot_control_stream()

    T = 0.5  # 总时间
    dt0 = [0.] * 18  # 假设 NMC 是一个定义好的常量，表示关节数量
    dt1 = [0.] * 18  # 创建一个 NMC 长度的列表，初始值为 0
    dt2 = [0.] * 18
    D2R = math.pi / 180.0
    # 填充 dt1 和 dt2 列表
    dt1[2] = 30 * D2R
    dt1[3] = -60 * D2R
    dt1[4] = 30 * D2R
    dt1[10] = -50 * D2R
    dt1[13] = 100 * D2R
    dt1[14] = 50.0 * D2R
    dt1[17] = 100 * D2R

    dt2[7] = 30 * D2R
    dt2[8] = -60 * D2R
    dt2[9] = 30 * D2R
    dt2[10] = 50 * D2R
    dt2[13] = 100 * D2R
    dt2[14] = -50.0 * D2R
    dt2[17] = 100 * D2R

    # 执行关节规划
    for i in range(10):
        print("wave round %d" % (i * 2 + 1))
        gBot.joint_plan(T, dt1)
        print("wave round %d" % (i * 2 + 2))
        gBot.joint_plan(T, dt2)
    print("return to zero")
    gBot.joint_plan(T, dt0)
    gBot.fox_stop()
