#!/usr/bin/env python3
"""
================================================================
  相机实时预览服务
================================================================

  通过 GalbotControl 订阅头部相机，Flask 提供 MJPEG 流。
  与推理服务可同时运行（推理用端口 6686/6687/6688，这里用 8080）。

  【用法】
    python camera_viewer.py
    python camera_viewer.py --port 8080

  【访问】
    浏览器: http://<机器人IP>:8080
================================================================
"""
import os
import sys
import time
import threading

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from args import Args
from galbot_control.galbot_control import GalbotControl
from tool.logger import LoggerManager
from tool.tool_shutdown import ShutdownTool
from flask import Flask, Response, jsonify
import argparse
import logging


class CameraViewer:
    """轻量相机预览（复用 GalbotControl.galbot_interface 的订阅）"""

    def __init__(self, galbot):
        self.galbot = galbot
        self.interface = galbot.galbot_interface
        self.logger = LoggerManager.get_logger()
        self.logger.info("[camera_viewer] 相机接口已就绪")

    def get_frame(self, key):
        """获取头部相机最新帧（JPEG 字节）"""
        if key == "head":
            return (
                (
                    bytes(self.interface._image_head_left)
                    if self.interface._image_head_left
                    else None
                ),
                self.interface._image_head_left_timestamp,
            )
        return None, 0.0

    def status(self):
        """返回头部相机连接状态"""
        now = time.time()
        return {
            "head": {
                "connected": self.interface._image_head_left is not None,
                "last_frame_age_s": (
                    round(now - self.interface._image_head_left_timestamp, 2)
                    if self.interface._image_head_left_timestamp > 0
                    else None
                ),
            },
        }

    def shutdown(self):
        self.logger.info("[camera_viewer] 相机预览服务关闭")
        self.galbot.shutdown()


app = Flask(__name__)
_viewer: CameraViewer = None

_PLACEHOLDER_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"


def _mjpeg_generator(key):
    # 约 30fps 推帧；无图像时发占位帧保持连接不断
    while True:
        frame, _ = _viewer.get_frame(key)
        jpeg = frame if frame else _PLACEHOLDER_JPEG
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.033)


@app.route("/stream/head")
def stream():
    return Response(
        _mjpeg_generator("head"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/snapshot/head")
def snapshot():
    frame, _ = _viewer.get_frame("head")
    return Response(frame if frame else _PLACEHOLDER_JPEG, mimetype="image/jpeg")


@app.route("/status")
def status():
    return jsonify(_viewer.status())


@app.route("/")
def index():
    s = _viewer.status()
    connected = s["head"]["connected"]
    badge = f'<span style="background:{"#22c55e" if connected else "#ef4444"};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{"连接" if connected else "离线"}</span>'
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>银河机器人 头部相机</title><style>*{{box-sizing:border-box;margin:0;padding:0}}html,body{{height:100%;overflow:hidden}}body{{background:#0f172a;color:#e2e8f0;font-family:sans-serif;display:flex;flex-direction:column}}header{{background:#1e293b;padding:8px 16px;border-bottom:1px solid #334155;flex-shrink:0;display:flex;justify-content:space-between;align-items:center}}h1{{font-size:16px;font-weight:600}}.view{{flex:1;display:flex;min-height:0}}img{{width:100%;height:100%;object-fit:contain;background:#0f172a;display:block}}.dot{{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:5px;animation:pulse 2s infinite}}@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}</style></head><body><header><h1><span class="dot"></span>头部相机</h1>{badge}</header><div class="view"><img src="/stream/head"></div></body></html>"""


if __name__ == "__main__":
    LoggerManager.get_logger()
    for listener in LoggerManager._queue_listeners.values():
        for h in listener.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="机器人头部相机实时预览服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    cli = parser.parse_args()

    print("=" * 60)
    print("银河机器人 头部相机预览服务")
    print("=" * 60)
    print("[初始化] GalbotControl...")

    args = Args()
    args.has_init_action = False
    galbot = GalbotControl(args)

    tool_shutdown = ShutdownTool()

    print("[等待] 传感器就绪...")
    t0 = time.time()
    while time.time() - t0 < 5:
        if (
            galbot.galbot_interface._joint_sensor_vla is not None
            and len(galbot.galbot_interface.pose_buffer) >= 2
        ):
            break
        time.sleep(0.05)
    print(f"[就绪] 传感器连接完成 ({time.time() - t0:.1f}s)")

    _viewer = CameraViewer(galbot)
    tool_shutdown.on_shutdown(_viewer.shutdown)

    print("=" * 60)

    app.run(host=cli.host, port=cli.port, threaded=True)