"""
================================================================
Replay 脚本
================================================================

【原理】
  从原始轨迹按固定间隔取关键帧（降采样），组成一段稀疏轨迹，
  一次性发给底层（一个大 mat），底层在关键帧之间自己平滑插值。

  例：3211帧原始数据，step=15 → 取约 214 个关键帧
      每个关键帧对应 step×frame_dt = 0.5s 的执行时间
      总时长不变，底层做插值，动作更丝滑。

  这和 reset 复位的机制完全一样——给"起点+终点+时间"，
  底层做平滑运动，不需要 Python 侧精确节拍发布。

【用法】
  python replay_downsample.py
  python replay_downsample.py --parquet xxx/episode_000000.parquet
  python replay_downsample.py --step 15
  python replay_downsample.py --speed 0.5
  python replay_downsample.py --start-frame 0 --end-frame 300

【参数】
  --parquet     parquet 路径（不传用 PARQUET_PATH）
  --fps         原始帧率，默认 30
  --speed       速度倍率，1.0原速 / 0.5慢放
  --step        降采样间隔（每隔多少帧取一帧），默认 15
                step=15 表示每 0.5s 一个关键帧（30fps×0.5s=15帧）
  --start-frame / --end-frame   原始帧区间
  --move-time   复位到第一帧的时间(秒)，默认 2
================================================================
"""

import sys
import os

# 抑制底层 C 库(NvMMLite 等)的 stderr；调试看报错时把下行注释掉
# os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from args import Args
from galbot_control.galbot_control import GalbotControl
from tool.tool_shutdown import ShutdownTool
from tool.logger import LoggerManager
import argparse
import logging
import time
import math
import numpy as np
import pandas as pd

PARQUET_PATH = "/home/galbot/vla_client/拿毛巾.parquet"


def action38_to_joints23(a):
    """数据集 38 维 action -> 机器人 23 维 init_pose 格式（夹爪 mm->m）"""
    a = np.asarray(a, dtype=float)
    return (
        list(a[16:21])  # 0:5  腰
        + list(a[21:23])  # 5:7  头
        + list(a[8:15])  # 7:14 左臂
        + [a[15] / 1000.0]  # 14   左爪
        + list(a[0:7])  # 15:22 右臂
        + [a[7] / 1000.0]  # 22   右爪
    )


def action38_to_chassis(a):
    """数据集 38 维 action -> 底盘 [x, y, yaw]"""
    a = np.asarray(a, dtype=float)
    return [a[31], a[32], math.atan2(a[33], a[34]) * 2]


def wait_ready(galbot, timeout=5.0):
    """等传感器就绪（关节 + 底盘），就绪即返回 True，超时返回 False。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        joint_ok = (
            galbot.galbot_interface._joint_sensor_vla is not None
            and len(galbot.galbot_interface.pose_buffer) >= 2
        )
        chassis_ok = (
            galbot.galbot_interface.last_chassis_pos is not None
            or galbot.args.enable_chassis <= 0
        )
        if joint_ok and chassis_ok:
            return True
        time.sleep(0.05)
    return False


def replay(galbot, df, fps, speed, start_frame, end_frame, move_time, step):
    n = len(df)
    end_frame = n if (end_frame is None or end_frame > n) else end_frame
    actions = list(df["action"].iloc[max(0, start_frame) : end_frame])
    if not actions:
        print("[replay] 没有可回放的帧")
        return

    # 降采样：每隔 step 帧取一帧，末尾强制包含最后一帧，确保动作完整执行到底
    indices = list(range(0, len(actions), step))
    if indices[-1] != len(actions) - 1:
        indices.append(len(actions) - 1)
    keyframes = [actions[i] for i in indices]

    frame_dt = 1.0 / fps / max(speed, 1e-6)
    key_dt = step * frame_dt
    total_time = (len(indices) - 1) * key_dt

    print(f"[replay] 原始帧数: {len(actions)}  step={step}")
    print(
        f"[replay] 降采样后关键帧: {len(keyframes)}  帧间隔: {key_dt:.3f}s  总时长: {total_time:.1f}s"
    )

    if not wait_ready(galbot, timeout=5.0):
        print("[replay] 传感器未就绪，无法回放")
        return

    # 复位到第一帧
    print(f"[replay] 复位到第一帧（{move_time}s）...")
    galbot.shutdown_event.clear()
    galbot.error_imformation = ""
    aim = np.array(
        [move_time]
        + action38_to_joints23(keyframes[0])
        + list(galbot.galbot_interface.chassis_pos)
    ).reshape(-1, 1)
    galbot.set_wholebody_angle_asynchronous(aim, 0.03, 0.004)
    if galbot.error_imformation:
        print(f"[replay] 复位失败: {galbot.error_imformation}")
        return
    print("[replay] 复位完成，开始降采样回放...")

    # mat 结构（交给底层一次性插值执行，与 reset 机制相同）：
    #   [:,0]   当前真实位姿起点（pose_buffer，夹爪单位 m）
    #   [1,j]   第 j 列的绝对时间戳（秒）
    #   [2:25,j]关节目标（夹爪经 action38_to_joints23 已 /1000 转 m）
    #   [25:28,j]底盘目标 [x, y, yaw]
    # 最终统一 *1000 把所有列夹爪从 m 换回 mm 再发给底层
    cur = list(galbot.galbot_interface.pose_buffer[1])  # 26维
    n_keys = len(keyframes)
    mat = np.zeros((2 + 26, 1 + n_keys))
    mat[2 : 2 + 26, 0] = cur[0:26]

    for j, a in enumerate(keyframes):
        mat[1, 1 + j] = (j + 1) * key_dt
        mat[2 : 2 + 23, 1 + j] = action38_to_joints23(a)
        mat[2 + 23 : 2 + 26, 1 + j] = action38_to_chassis(a)

    # 夹爪 m → mm：pose_buffer 里夹爪也是 m，统一换算保证全列单位为 mm
    mat[2 + 14, :] *= 1000  # 左爪
    mat[2 + 22, :] *= 1000  # 右爪

    print(f"[replay] 发布轨迹（{n_keys} 个关键帧，预计 {total_time:.1f}s）...")
    galbot.embosa_vla_publisher.pub_mat(mat)

    # 等轨迹执行完（可被 Ctrl+C 打断）
    t_end = time.perf_counter() + total_time
    while time.perf_counter() < t_end:
        if galbot.shutdown_event.is_set():
            print("\n[replay] 被中断")
            break
        remain = t_end - time.perf_counter()
        sys.stdout.write(f"\r[replay] 剩余 {remain:5.1f}s")
        sys.stdout.flush()
        time.sleep(0.2)

    sys.stdout.write("\n")
    print("[replay] 回放结束")


if __name__ == "__main__":
    LoggerManager.get_logger()
    for listener in LoggerManager._queue_listeners.values():
        for h in listener.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="降采样 Replay：每隔 step 帧取关键帧，一次性发给底层插值执行"
    )
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--fps", type=int, default=30, help="原始帧率，默认30")
    parser.add_argument("--speed", type=float, default=1.0, help="速度倍率")
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="降采样间隔（每隔多少帧取一帧），默认15（=0.5s/关键帧）",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--move-time", type=float, default=2)
    cli = parser.parse_args()

    pq = cli.parquet or PARQUET_PATH
    if not os.path.exists(pq):
        print(f"[replay] parquet 不存在: {pq}")
        sys.exit(1)
    print(f"[replay] 数据文件: {pq}")
    df = pd.read_parquet(pq)
    print(
        f"[replay] 总帧数: {len(df)}  fps={cli.fps} speed={cli.speed} step={cli.step}"
    )

    galbot = GalbotControl(Args())
    tool_shutdown = ShutdownTool()
    tool_shutdown.on_shutdown(galbot.shutdown)

    # 等底层连接就绪，连好了立即继续，最多等 3 秒兜底
    print("[replay] 等待底层连接就绪...")
    if wait_ready(galbot, timeout=3.0):
        print("[replay] 连接就绪，开始 replay")
    else:
        print("[replay] 等待超时，继续执行")

    replay(
        galbot,
        df,
        cli.fps,
        cli.speed,
        cli.start_frame,
        cli.end_frame,
        cli.move_time,
        cli.step,
    )