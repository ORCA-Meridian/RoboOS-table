"""
Galbot-1 MCP Skill Server
=========================
作为 RoboOS slaver 的工具服务端，暴露清理桌面任务的四个步骤。

执行逻辑：
  - Step 1/2/4 (推理步骤): 调 HTTP 接口启动推理 → VLM 轮询相机判断完成 → 调 stop → 返回
  - Step 3 (replay): 先调 stop 确保推理停止 → 调 replay 接口 → 轮询 /api/status 等结束 → 返回
  每个工具都是阻塞的，完成后才返回，Slaver 才会调下一个工具。

【启动】
  python skill.py
  python skill.py --config config.yaml
"""

import argparse
import base64
import time
import logging
import os
import sys
from typing import Optional

import requests
import yaml
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH)
_cli, _ = parser.parse_known_args()

CFG = _load_config(_cli.config)

_GALBOT = CFG["galbot_server"]
_CAM = CFG["camera_server"]
_VLM = CFG["vlm"]
_REPLAY = CFG["replay"]

GALBOT_BASE = "http://{}:{}".format(_GALBOT["host"], _GALBOT["port"])
CAM_SNAPSHOT_URL = "http://{}:{}{}".format(_CAM["host"], _CAM["port"], _CAM["snapshot_path"])

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, CFG.get("logger", {}).get("level", "INFO")),
    format="%(asctime)s [galbot-skill] %(levelname)s %(message)s",
)
log = logging.getLogger("galbot-skill")

# ---------------------------------------------------------------------------
# 底层 HTTP 工具
# ---------------------------------------------------------------------------

def _galbot_post(path: str, body: Optional[dict] = None, timeout: int = None) -> dict:
    """向 galbot HTTP 服务发 POST，返回 JSON。"""
    r = requests.post(
        GALBOT_BASE + path,
        json=body or {},
        timeout=timeout or _GALBOT["timeout"],
    )
    r.raise_for_status()
    return r.json()


def _galbot_get(path: str) -> dict:
    """向 galbot HTTP 服务发 GET，返回 JSON。"""
    r = requests.get(GALBOT_BASE + path, timeout=_GALBOT["timeout"])
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 相机快照
# ---------------------------------------------------------------------------

def _fetch_snapshot_b64() -> Optional[str]:
    """从相机服务抓一帧头部图像，返回 base64 JPEG。失败返回 None。"""
    try:
        r = requests.get(CAM_SNAPSHOT_URL, timeout=5)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8")
    except Exception as e:
        log.warning("[snapshot] 抓帧失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# VLM 完成判断
# ---------------------------------------------------------------------------

def _vlm_check_done(b64_image: str, prompt: str) -> bool:
    """调用 VLM 判断当前帧任务是否完成，只有回答 yes 才返回 True。"""
    from openai import OpenAI
    client = OpenAI(api_key=_VLM["api_key"], base_url=_VLM["api_base"])
    try:
        resp = client.chat.completions.create(
            model=_VLM["model"],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64_image},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=16,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip().lower()
        log.info("[vlm] 判断结果: '%s'", answer)
        return answer.startswith("yes")
    except Exception as e:
        log.warning("[vlm] 调用失败: %s", e)
        return False


# ---------------------------------------------------------------------------
# 核心等待逻辑
# ---------------------------------------------------------------------------

def _poll_until_done(step_key: str) -> str:
    """
    VLM 轮询相机图像直到任务完成或超时。
    完成或超时后都调 /api/stop 停止推理，然后返回。
    """
    prompt = _VLM["completion_prompts"][step_key]
    poll_interval = float(_VLM["poll_interval"])
    max_polls = int(_VLM["max_polls"])

    log.info("[%s] 开始 VLM 轮询，间隔 %.1fs，最多 %d 次", step_key, poll_interval, max_polls)

    for i in range(1, max_polls + 1):
        time.sleep(poll_interval)
        b64 = _fetch_snapshot_b64()
        if b64 is None:
            log.warning("[%s] 第 %d 次：抓帧失败，跳过", step_key, i)
            continue
        if _vlm_check_done(b64, prompt):
            log.info("[%s] VLM 判断完成（第 %d 次），调用 stop", step_key, i)
            try:
                _galbot_post("/api/stop")
            except Exception as e:
                log.warning("[%s] stop 调用失败: %s", step_key, e)
            return "Step '{}' completed after {} VLM polls.".format(step_key, i)
        log.info("[%s] 第 %d/%d 次：未完成，继续等待", step_key, i, max_polls)

    log.warning("[%s] 超过最大轮询次数 (%d)，强制 stop", step_key, max_polls)
    try:
        _galbot_post("/api/stop")
    except Exception as e:
        log.warning("[%s] stop 调用失败: %s", step_key, e)
    return "Step '{}' timed out after {} VLM polls, force stopped.".format(step_key, max_polls)


def _wait_task_finish(step_label: str) -> str:
    """
    轮询 /api/status 直到 running=False。
    用于 replay 等不需要 VLM 的步骤（replay 执行完后 running 自动变 False）。
    """
    poll_interval = float(_GALBOT["task_poll_interval"])
    max_wait = int(_GALBOT["task_max_wait"])
    deadline = time.time() + max_wait

    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            data = _galbot_get("/api/status").get("data", {})
            if not data.get("running", True):
                success = data.get("success", False)
                msg = data.get("message", "")
                log.info("[%s] 任务结束: success=%s, msg=%s", step_label, success, msg)
                return "Step '{}' finished: success={}, {}".format(step_label, success, msg)
        except Exception as e:
            log.warning("[%s] 轮询 status 失败: %s", step_label, e)

    return "Step '{}' timed out waiting for completion.".format(step_label)


# ---------------------------------------------------------------------------
# MCP 工具定义
# ---------------------------------------------------------------------------

mcp = FastMCP(name="galbot-1", stateless_http=True, host="0.0.0.0", port=8000)


@mcp.tool()
async def pick_bag() -> str:
    """
    [STEP 1 of 4] Pick up the trash bag from the floor and place it onto the table.
    MUST be called first before any other step.
    The robot crouches down, grasps the bag from the floor, and places it on the table.
    Blocks until VLM confirms the bag is on the table, then stops inference.
    After this returns success, call bag_large_items next.
    """
    log.info("[pick_bag] 启动 step1 推理...")
    try:
        _galbot_post("/api/pick_bag")
    except Exception as e:
        return "Failed to start pick_bag: {}".format(e)

    # pick_bag 有复位动作，等复位完成后再开始 VLM 轮询
    log.info("[pick_bag] 等待机器人复位完成 (3s)...")
    time.sleep(3.0)

    result = _poll_until_done("pick_bag")
    log.info("[pick_bag] 结果: %s", result)
    return result


@mcp.tool()
async def bag_large_items() -> str:
    """
    [STEP 2 of 4] Clear large items from the table into the trash bag.
    MUST be called after pick_bag succeeds.
    The robot picks up bottles, cans, snack bags, paper balls and other trash
    and places them into the trash bag on the table.
    IMPORTANT: The white takeout box (lobster container) MUST remain on the table.
    Blocks until VLM confirms all large items are bagged, then stops inference.
    After this returns success, call replay_towel next.
    """
    log.info("[bag_large_items] 启动 step2 推理...")
    try:
        _galbot_post("/api/bag_large_items")
    except Exception as e:
        return "Failed to start bag_large_items: {}".format(e)

    result = _poll_until_done("bag_large_items")
    log.info("[bag_large_items] 结果: %s", result)
    return result


@mcp.tool()
async def replay_towel(
    parquet_path: str = None,
    speed: float = 1.0,
    step: int = 15,
) -> str:
    """
    [STEP 3 of 4] Replay the pre-recorded trajectory to fetch the towel onto the table.
    MUST be called after bag_large_items succeeds.
    This uses a recorded motion file — no model inference. It first stops any running
    inference, then replays the trajectory so the robot fetches the towel and lays it
    on the table. Blocks until the replay finishes naturally (no VLM needed here).
    After this returns success, call sweep_trash next.

    Args:
        parquet_path: Absolute path to the .parquet file on the robot. Uses config default if omitted.
        speed: Playback speed multiplier (default 1.0 = normal speed).
        step: Downsampling interval in frames (default 15 = one keyframe per 0.5 s).
    """
    pq = parquet_path or _REPLAY["parquet_path"]

    # replay 前必须先 stop，否则 robot_server 内部也会等底层停止
    # 这里提前发 stop 让上层 HTTP 服务知道要停，robot_server 里还会二次确认底层停止
    log.info("[replay_towel] 先调 stop 确保推理停止...")
    try:
        _galbot_post("/api/stop")
    except Exception as e:
        log.warning("[replay_towel] stop 调用失败（继续）: %s", e)
    time.sleep(1.0)  # 给底层 1s 余量真正停下来

    log.info("[replay_towel] 启动 replay，parquet=%s speed=%.1f step=%d", pq, speed, step)
    try:
        _galbot_post(
            "/api/replay_downsample",
            body={
                "parquet_path": pq,
                "fps": _REPLAY["fps"],
                "speed": speed,
                "step": step,
                "no_reset": _REPLAY["no_reset"],
            },
        )
    except Exception as e:
        return "Failed to start replay_towel: {}".format(e)

    # replay 结束后 running 自动变 False，轮询等待即可
    result = _wait_task_finish("replay_towel")
    log.info("[replay_towel] 结果: %s", result)
    return result


@mcp.tool()
async def sweep_trash() -> str:
    """
    [STEP 4 of 4] Use the towel to sweep lobster debris into the white takeout box,
    then place the box into the trash bag. This is the FINAL step.
    MUST be called after replay_towel succeeds (towel must already be on the table).
    The robot wipes lobster shell fragments into the white container, then bags it.
    Blocks until VLM confirms the table is clean and the box is bagged, then stops inference.
    After this returns success the entire table-clearing task is complete.
    """
    log.info("[sweep_trash] 启动 step4 推理...")
    try:
        _galbot_post("/api/sweep_trash")
    except Exception as e:
        return "Failed to start sweep_trash: {}".format(e)

    result = _poll_until_done("sweep_trash")
    log.info("[sweep_trash] 结果: %s", result)
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Galbot-1 MCP Skill Server")
    log.info("  galbot server : %s", GALBOT_BASE)
    log.info("  camera        : %s", CAM_SNAPSHOT_URL)
    log.info("  VLM model     : %s @ %s", _VLM["model"], _VLM["api_base"])
    log.info("  tools         : pick_bag / bag_large_items / replay_towel / sweep_trash")
    log.info("=" * 60)
    mcp.run(transport="streamable-http")
