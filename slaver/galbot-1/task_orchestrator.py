"""
Galbot-1 长程任务编排器
=======================
独立于 RoboOS master/slaver 框架，直接在工作站或机器人本机运行。
按顺序执行清理桌面的四个步骤：

  Step 1  pick_bag        — 模型推理：从地面拾起垃圾袋放上桌
  Step 2  bag_large_items — 模型推理：将桌面大件垃圾装入袋（龙虾饭盒除外）
  Step 3  replay_towel    — Replay：播放拿毛巾轨迹，毛巾放上桌
  Step 4  sweep_trash     — 模型推理：用毛巾将龙虾擦进饭盒，装入垃圾袋

步骤 1/2/4 通过 VLM 轮询头部相机快照判断完成，完成后调用 /api/stop。
步骤 3 通过轮询 /api/status 等待 replay 自然结束。

【用法】
  python task_orchestrator.py
  python task_orchestrator.py --config config.yaml
  python task_orchestrator.py --start-step 2     # 从第2步开始（断点续跑）
  python task_orchestrator.py --only-step 3      # 只跑第3步
"""

import argparse
import base64
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Callable
import requests
import yaml

from clients.camera_client import CameraClient
from clients.vlm_openai_client import OpenAICompatibleVLMClient


_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logger(cfg: dict) -> logging.Logger:
    logger_cfg = cfg.get("logger", {})
    log_file = logger_cfg.get("file", ".logs/galbot1_orchestrator.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [orchestrator] %(levelname)s %(message)s")
    level = getattr(logging, logger_cfg.get("level", "INFO"))
    h_stream = logging.StreamHandler(sys.stdout)
    h_stream.setFormatter(fmt)
    h_file = logging.FileHandler(log_file, encoding="utf-8")
    h_file.setFormatter(fmt)
    log = logging.getLogger("galbot-orchestrator")
    log.setLevel(level)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    log.addHandler(h_stream)
    log.addHandler(h_file)
    return log


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------

class GalbotClient:
    """与 robot_server_clear_table.py 通信的 HTTP 客户端。"""

    def __init__(self, cfg: dict, log: logging.Logger):
        gs = cfg["galbot_server"]
        self.base = "http://{}:{}".format(gs["host"], gs["port"])
        self.timeout = gs["timeout"]
        self.poll_interval = gs["task_poll_interval"]
        self.max_wait = gs["task_max_wait"]
        self.log = log

    def post(self, path: str, body: dict = None) -> dict:
        r = requests.post(self.base + path, json=body or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get(self, path: str) -> dict:
        r = requests.get(self.base + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def stop(self):
        try:
            self.post("/api/stop")
            self.log.info("[client] stop 指令已发送")
        except Exception as e:
            self.log.warning("[client] stop 失败: %s", e)

    def reset(self, label: str = "reset") -> bool:
        """调用 /api/reset 触发复位，然后轮询 /api/status 等待复位完成。"""
        try:
            self.post("/api/reset")
            self.log.info("[%s] 复位指令已发送", label)
        except Exception as e:
            self.log.warning("[%s] 复位指令发送失败: %s", label, e)
            return False
        ok, msg = self.wait_task_finish(label)
        self.log.info("[%s] 复位结果: success=%s, msg=%s", label, ok, msg)
        return ok

    def wait_task_finish(self, label: str) -> tuple:
        """轮询 /api/status 直到任务结束，返回 (success, message)。"""
        deadline = time.time() + self.max_wait
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            try:
                data = self.get("/api/status").get("data", {})
                if not data.get("running", True):
                    ok = bool(data.get("success", False))
                    msg = data.get("message", "")
                    self.log.info("[%s] 任务结束: success=%s, msg=%s", label, ok, msg)
                    return ok, msg
            except Exception as e:
                self.log.warning("[%s] 轮询 status 出错: %s", label, e)
        self.log.warning("[%s] 等待超时 (%ds)", label, self.max_wait)
        return False, "timeout"


# ---------------------------------------------------------------------------
# VLM 完成判断
# ---------------------------------------------------------------------------

class VLMJudge:
    """用 VLM 轮询头部相机图像判断当前步骤是否完成。"""

    def __init__(self, cfg, log):
        vlm = cfg["vlm"]
        self.camera = CameraClient(cfg["camera_server"], log)
        self.vlm_client = OpenAICompatibleVLMClient(vlm, log)
        self.poll_interval = float(vlm["poll_interval"])
        self.max_polls = int(vlm["max_polls"])
        self.prompts = vlm["completion_prompts"]
        self.post_done_sleep = vlm.get("post_done_sleep", {})
        self.log = log
        self.example_imgs = self._load_example_images()

    def _load_example_images(self) -> dict:
        """预加载三个任务的成功示例图（base64），启动时读一次，避免重复 IO。"""
        img_dir = os.path.join(os.path.dirname(__file__), "img")
        mapping = {
            "pick_bag":        "pick_bag.png",
            "bag_large_items": "bag_large_items.png",
            "sweep_trash":     "sweep_trash.png",
        }
        result = {}
        for key, fname in mapping.items():
            path = os.path.join(img_dir, fname)
            try:
                with open(path, "rb") as f:
                    result[key] = base64.b64encode(f.read()).decode("utf-8")
                self.log.info("[vlm] 示例图加载成功: %s", fname)
            except Exception as e:
                self.log.warning("[vlm] 示例图加载失败 %s: %s", fname, e)
                result[key] = None
        return result


    def _snapshot_b64(self):
        return self.camera.snapshot_b64("vlm")

    def _ask(self, b64, prompt, step_key=None):
        try:
            content = []
            example_b64 = self.example_imgs.get(step_key) if step_key else None
            if example_b64:
                content += [
                    {"type": "text", "text": "Here is an example image showing the COMPLETED state of this step for your reference:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + example_b64}},
                    {"type": "text", "text": "Now look at the current camera image and answer the question below:"},
                ]
            content += [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": prompt.strip()},
            ]
            answer = self.vlm_client.judge_completion(content)
            self.log.info("[vlm] 回答: '%s'", answer)
            return answer.startswith("yes")
        except Exception as e:
            self.log.warning("[vlm] 调用出错: %s", e)
            return False

    def poll_until_done(self, step_key):
        """轮询到 VLM 判断完成或超过 max_polls，返回 (done, polls_used)。"""
        import time
        prompt = self.prompts[step_key]
        self.log.info("[%s] 开始 VLM 轮询，间隔 %.1fs，最多 %d 次",
                      step_key, self.poll_interval, self.max_polls)
        for i in range(1, self.max_polls + 1):
            time.sleep(self.poll_interval)
            b64 = self._snapshot_b64()
            if b64 is None:
                self.log.warning("[%s] 第 %d 次：抓帧失败，跳过", step_key, i)
                continue
            if self._ask(b64, prompt, step_key=step_key):
                self.log.info("[%s] 第 %d 次：VLM 判断完成", step_key, i)
                delay = float(self.post_done_sleep.get(step_key, 0.0))
                if delay > 0:
                    self.log.info("[%s] 等待 %.1fs 后结束轮询 ...", step_key, delay)
                    time.sleep(delay)
                return True, i
            self.log.info("[%s] 第 %d/%d 次：未完成", step_key, i, self.max_polls)
        self.log.warning("[%s] 超过最大轮询次数，视为超时", step_key)
        return False, self.max_polls


# ---------------------------------------------------------------------------
# 步骤数据结构
# ---------------------------------------------------------------------------



@dataclass
class StepResult:
    step_id: int
    name: str
    success: bool
    message: str
    duration: float


@dataclass
class Step:
    step_id: int
    name: str
    label: str
    run: Callable


# ---------------------------------------------------------------------------
# 编排器
# ---------------------------------------------------------------------------

class TableClearOrchestrator:
    """清理桌面长程任务编排器，按顺序执行四个步骤。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = _setup_logger(cfg)
        self.client = GalbotClient(cfg, self.log)
        self.vlm = VLMJudge(cfg, self.log)
        self.steps = self._build_steps()

    def _infer_step(self, step_id, name, api_path, pre_sleep=0.0):
        import time
        t0 = time.time()
        self.log.info("=" * 50)
        self.log.info("Step %d / %s", step_id, name)
        self.log.info("=" * 50)
        start_retries = 3
        for attempt in range(1, start_retries + 1):
            try:
                self.client.post(api_path)
                break
            except Exception as e:
                self.log.warning("[%s] 启动失败 attempt %d/%d: %s", name, attempt, start_retries, e)
                if attempt == start_retries:
                    return StepResult(step_id, name, False, "启动失败: " + str(e), time.time() - t0)
                time.sleep(2.0)
        if pre_sleep > 0:
            self.log.info("[%s] 等待初始化 %.1fs ...", name, pre_sleep)
            time.sleep(pre_sleep)
        done, polls = self.vlm.poll_until_done(name)
        self.client.stop()
        return StepResult(step_id, name, done,
                          "VLM polls={}, done={}".format(polls, done),
                          time.time() - t0)

    def _run_pick_bag(self):
        result = self._infer_step(1, "pick_bag", "/api/pick_bag", pre_sleep=3.0)
        if result.success:
            self.log.info("[pick_bag] Step 1 完成，开始复位...")
            self.client.reset("pick_bag_reset")
        return result

    def _run_bag_large_items(self):
        return self._infer_step(2, "bag_large_items", "/api/bag_large_items")

    def _run_replay_towel(self):
        import time
        name = "replay_towel"
        t0 = time.time()
        self.log.info("=" * 50)
        self.log.info("Step 3 / replay_towel — Replay 拿毛巾轨迹")
        self.log.info("=" * 50)
        rc = self.cfg["replay"]
        try:
            self.client.post("/api/replay_downsample", body={
                "parquet_path": rc["parquet_path"],
                "fps": rc["fps"],
                "speed": rc["speed"],
                "step": rc["step"],
                "no_reset": rc["no_reset"],
            })
        except Exception as e:
            return StepResult(3, name, False, "启动失败: " + str(e), time.time() - t0)
        ok, msg = self.client.wait_task_finish(name)
        if ok:
            self.log.info("[replay_towel] Replay 完成，开始复位...")
            self.client.reset("replay_towel_reset")
        return StepResult(3, name, ok, msg, time.time() - t0)

    def _run_sweep_trash(self):
        result = self._infer_step(4, "sweep_trash", "/api/sweep_trash")
        if result.success:
            self.log.info("[sweep_trash] Step 4 完成，开始复位...")
            self.client.reset("sweep_trash_reset")
        return result

    def _build_steps(self):
        return [
            Step(1, "pick_bag",        "Step 1 — 拾起垃圾袋放上桌", self._run_pick_bag),
            Step(2, "bag_large_items", "Step 2 — 桌面大件装袋",      self._run_bag_large_items),
            Step(3, "replay_towel",    "Step 3 — Replay 拿毛巾",     self._run_replay_towel),
            Step(4, "sweep_trash",     "Step 4 — 毛巾擦龙虾装袋",    self._run_sweep_trash),
        ]

    def run(self, start_step=1, only_step=None):
        import time
        self.log.info("*" * 60)
        self.log.info("Galbot-1 清理桌面任务开始  start_step=%d  only_step=%s",
                      start_step, only_step)
        self.log.info("*" * 60)
        total_t0 = time.time()
        results = []

        steps_to_run = (
            [s for s in self.steps if s.step_id == only_step]
            if only_step
            else [s for s in self.steps if s.step_id >= start_step]
        )
        if not steps_to_run:
            self.log.error("没有可执行的步骤，检查 --start-step / --only-step 参数")
            return False

        all_ok = True
        for step in steps_to_run:
            self.log.info(">>> 开始执行: %s", step.label)
            result = step.run()
            results.append(result)
            flag = "成功" if result.success else "失败/超时"
            self.log.info("<<< %s — %s  耗时 %.1fs  %s",
                          step.label, flag, result.duration, result.message)
            if not result.success:
                all_ok = False
                self.log.error("步骤 %d (%s) 未成功，任务终止", step.step_id, step.name)
                break

        total_dur = time.time() - total_t0
        self.log.info("*" * 60)
        self.log.info("任务%s，总耗时 %.1fs",
                      "全部完成" if all_ok else "中途失败", total_dur)
        for r in results:
            self.log.info("  %s Step %d [%s]  %.1fs  %s",
                          "OK" if r.success else "NG", r.step_id, r.name,
                          r.duration, r.message)
        self.log.info("*" * 60)
        return all_ok

# ---------------------------------------------------------------------------
# VLM 驱动编排器（function calling 模式）
# ---------------------------------------------------------------------------

class VLMDrivenOrchestrator:
    """
    完全由 VLM function calling 驱动的任务编排器。
    每轮拍一张当前相机图，同时提供三个阶段成功示例图，由 VLM 决定下一步调哪个函数。
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "run_pick_bag",
                "description": (
                    "Call this if the trash bag is not clearly on the right side of the table. "
                    "This starts the robot action to pick up the trash bag from the floor and place it on the table."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_bag_large_items",
                "description": (
                    "Call this if the trash bag is already on the table, but loose trash items "
                    "such as bottles, cans, cartons, snack bags, paper balls, or miscellaneous trash "
                    "are still visible on the table. Do not call this for red lobster shells inside the white box."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_replay_towel",
                "description": (
                    "Call this after loose trash has been bagged and before sweeping lobster shells. "
                    "This replays the towel-fetching trajectory to place the towel on the table."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_sweep_trash",
                "description": (
                    "Call this when the towel should be used to sweep red lobster shells and residue "
                    "into the white takeout box, then bag the white box and towel."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "continue_current_action",
                "description": (
                    "Call this when the current robot action should continue and no new action should be started. "
                    "This tool does nothing; the orchestrator will wait and observe again."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_current_action",
                "description": (
                    "Call this when the current robot action has achieved its goal and should be stopped. "
                    "After stopping, the orchestrator will observe again and decide the next action."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_completed",
                "description": (
                    "Call this only when the white takeout box is no longer on the table, "
                    "the trash bag is on the right side of the table, and the table surface is mostly clean "
                    "with no visible red lobster shells or residue."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_failed",
                "description": (
                    "Call this if the current state is unsafe, impossible to determine, or the robot cannot proceed."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    SYSTEM_PROMPT = (
        "You are a robot task controller for a table-cleaning task. "
        "You must decide the next robot function call by comparing the CURRENT camera image "
        "with reference images of completed stages. "
        "The task has this intended order: "
        "Step 1: pick up the trash bag from the floor and place it on the right side of the table. "
        "Step 2: put loose trash items into the trash bag, while keeping the white lobster takeout box on the table. "
        "Step 3: fetch the towel using replay trajectory. "
        "Step 4: use the towel to sweep red lobster shells and residue into the white takeout box, "
        "then put the white box and towel into the trash bag. "
        "Always call exactly one function. "
        "If an action is running and not finished, call continue_current_action. "
        "If an action is running and its goal is achieved, call stop_current_action. "
        "Do not skip a required step unless the current image clearly shows that step is already completed."
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = _setup_logger(cfg)
        self.vlm_cfg = cfg["vlm"]

        # 加载 vlm_driven 专用 config（提示词、规则、重试参数）
        _vd_path = os.path.join(os.path.dirname(__file__), "vlm_driven_config.yaml")
        with open(_vd_path, "r", encoding="utf-8") as _f:
            _vd = yaml.safe_load(_f)
        self.vd_cfg = _vd.get("vlm_driven", {})

        self.decision_timeout = float(self.vd_cfg.get("decision_timeout", 60))
        self.api_retries = int(self.vd_cfg.get("api_retries", 2))
        self.api_retry_sleep = float(self.vd_cfg.get("api_retry_sleep", 2.0))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.camera = CameraClient(cfg["camera_server"], self.log)
        self.vlm_client = OpenAICompatibleVLMClient(self.vlm_cfg, self.log)
        self.max_rounds = int(self.vd_cfg.get("max_rounds", 20))
        self.poll_interval = float(self.vlm_cfg["poll_interval"])
        self.example_imgs = self._load_example_images()
        self.log.info(
            "[vlm-driven] 可用 tool functions: %s",
            ", ".join(t["function"]["name"] for t in self.TOOLS),
        )
        
        self.current_action = None
        self.client_galbot = GalbotClient(cfg, self.log)

    def _load_example_images(self) -> dict:
        img_dir = os.path.join(os.path.dirname(__file__), "img")
        mapping = {
            "pick_bag": "pick_bag.png",
            "bag_large_items": "bag_large_items.png",
            "sweep_trash": "sweep_trash.png",
        }
        result = {}
        for key, fname in mapping.items():
            path = os.path.join(img_dir, fname)
            try:
                with open(path, "rb") as f:
                    result[key] = base64.b64encode(f.read()).decode("utf-8")
                self.log.info("[vlm-driven] 示例图加载成功: %s", fname)
            except Exception as e:
                self.log.warning("[vlm-driven] 示例图加载失败 %s: %s", fname, e)
                result[key] = None
        return result

    def _snapshot_b64(self):
        return self.camera.snapshot_b64("vlm-driven")

    def _start_action(self, fn: str) -> bool:
        """启动一个机器人动作。VLM-driven 模式下只启动，不在这里做 yes/no 完成判断。"""
        api_map = {
            "run_pick_bag": "/api/pick_bag",
            "run_bag_large_items": "/api/bag_large_items",
            "run_sweep_trash": "/api/sweep_trash",
        }
        if fn == "run_replay_towel":
            rc = self.cfg["replay"]
            body = {
                "parquet_path": rc["parquet_path"],
                "fps": rc["fps"],
                "speed": rc["speed"],
                "step": rc["step"],
            }
            self.client_galbot.post("/api/replay_downsample", body=body)
            self.current_action = fn
            return True
        api_path = api_map.get(fn)
        if not api_path:
            return False
        self.client_galbot.post(api_path)
        self.current_action = fn
        return True

    def _stop_current_action(self):
        """停止当前动作。如果该动作在 vlm_driven_config.yaml 配置了 stop_delays，先等待再发 stop。"""
        action = self.current_action
        try:
            delay = float(self.vd_cfg.get("stop_delays", {}).get(action or "", 0.0))
            if delay > 0:
                self.log.info("[vlm-driven] stop_current_action: 等待 %.1fs 后停止 %s ...", delay, action)
                time.sleep(delay)
            self.client_galbot.stop()
        finally:
            self.current_action = None

    def _build_content(self, b64: str) -> list:
        """根据 vlm_driven_config.yaml 构造发给 VLM 的 content 列表。"""
        captions = self.vd_cfg.get("example_captions", {})
        completion_rules = self.vd_cfg.get("completion_rules", {})
        current_frame_caption = self.vd_cfg.get("current_frame_caption", "Image 4: CURRENT camera view.")
        decision_rules_tpl = self.vd_cfg.get("decision_rules", "")

        # 把 completion_rules 内容拼进 decision_rules 前面
        rules_block = ""
        for step_key in ("pick_bag", "bag_large_items", "replay_towel", "sweep_trash"):
            rule = completion_rules.get(step_key, "")
            if rule:
                rules_block += f"\n{rule.strip()}\n"

        decision_text = (rules_block.strip() + "\n\n" + decision_rules_tpl).format(
            current_action=self.current_action or "none"
        )

        content = []

        # Image 1: pick_bag example
        if self.example_imgs.get("pick_bag"):
            content += [
                {"type": "text", "text": captions.get("pick_bag", "Image 1: completed example of pick_bag.").strip()},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + self.example_imgs["pick_bag"]}},
            ]

        # Image 2: bag_large_items example
        if self.example_imgs.get("bag_large_items"):
            content += [
                {"type": "text", "text": captions.get("bag_large_items", "Image 2: completed example of bag_large_items.").strip()},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + self.example_imgs["bag_large_items"]}},
            ]

        # Image 3: sweep_trash / final clean example
        if self.example_imgs.get("sweep_trash"):
            content += [
                {"type": "text", "text": captions.get("sweep_trash", "Image 3: completed example of the final cleaned state.").strip()},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + self.example_imgs["sweep_trash"]}},
            ]

        # Image 4: current camera frame
        content += [
            {"type": "text", "text": current_frame_caption.strip()},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": decision_text.strip()},
        ]

        return content

    def _decide_sync(self, b64: str) -> str:
        """同步调用 VLM function calling，返回 function name。失败时重试，重试耗尽返回 continue_current_action。"""

        system_prompt = self.vd_cfg.get("system_prompt", self.SYSTEM_PROMPT).strip()
        content = self._build_content(b64)

        last_err = None
        for attempt in range(1, self.api_retries + 2):
            try:
                self.log.info("[vlm-driven] 请求 VLM function calling 决策 (attempt %d)...", attempt)
                fn, message_content = self.vlm_client.decide(
                    system_prompt=system_prompt,
                    content=content,
                    tools=self.TOOLS,
                )
                if fn:
                    return fn
                self.log.warning("[vlm-driven] 模型未返回 tool_calls，content=%s", message_content)
                return "continue_current_action"
            except Exception as e:
                last_err = e
                self.log.warning("[vlm-driven] VLM 调用失败 (attempt %d): %s", attempt, e)
                if attempt <= self.api_retries:
                    time.sleep(self.api_retry_sleep)

        self.log.warning("[vlm-driven] 全部重试失败，本轮跳过: %s", last_err)
        return "continue_current_action"

    def _decide(self, b64: str) -> str:
        """异步调用 VLM 决策，带 timeout。超时或失败返回 continue_current_action，避免中断任务。"""
        future = self.executor.submit(self._decide_sync, b64)
        try:
            return future.result(timeout=self.decision_timeout)
        except TimeoutError:
            self.log.warning("[vlm-driven] VLM 决策超时 %.1fs，本轮跳过", self.decision_timeout)
            return "continue_current_action"
        except Exception as e:
            self.log.warning("[vlm-driven] VLM 异步决策异常，本轮跳过: %s", e)
            return "continue_current_action"

    def run(self):
        self.log.info("=" * 60)
        self.log.info("VLM function calling 驱动模式启动，最多 %d 轮", self.max_rounds)
        self.log.info("=" * 60)
        t0 = time.time()
        try:
            for round_i in range(1, self.max_rounds + 1):
                self.log.info("[round %d] 抓取当前帧...", round_i)
                time.sleep(self.poll_interval)
                b64 = self._snapshot_b64()
                if b64 is None:
                    self.log.warning("[round %d] 抓帧失败，跳过", round_i)
                    continue

                fn = self._decide(b64)

                if fn == "task_completed":
                    self.log.info("VLM 判断任务全部完成，总耗时 %.1fs", time.time() - t0)
                    return True

                if fn == "task_failed":
                    self.log.error("VLM 判断任务失败，总耗时 %.1fs", time.time() - t0)
                    return False

                if fn == "continue_current_action":
                    self.log.info("[round %d] tool_result: name=continue_current_action success=True message=no-op, wait for next observation", round_i)
                    continue

                if fn == "stop_current_action":
                    self.log.info("[round %d] 执行 tool function: stop_current_action", round_i)
                    self._stop_current_action()
                    self.log.info("[round %d] tool_result: name=stop_current_action success=True message=current action stopped", round_i)
                    continue

                if fn.startswith("run_"):
                    if self.current_action:
                        self.log.info("[round %d] 当前动作 %s 仍在运行，先 stop 再启动 %s", round_i, self.current_action, fn)
                        self._stop_current_action()
                    self.log.info("[round %d] 执行 tool function: %s", round_i, fn)
                    try:
                        ok = self._start_action(fn)
                    except Exception as e:
                        self.log.error("[round %d] %s 启动失败: %s", round_i, fn, e)
                        return False
                    self.log.info(
                        "[round %d] tool_result: name=%s success=%s message=started current_action=%s",
                        round_i,
                        fn,
                        ok,
                        self.current_action,
                    )
                    if not ok:
                        self.log.error("[round %d] 未知或无法启动 function: %s", round_i, fn)
                        return False
                    continue

                self.log.error("[round %d] 未知 function: %s", round_i, fn)
                return False
            self.log.warning("超过最大轮数 %d，任务未完成", self.max_rounds)
            return False
        finally:
            self.executor.shutdown(wait=False)
# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Galbot-1 清理桌面长程任务编排器")
    ap.add_argument("--config", default=_DEFAULT_CONFIG, help="配置文件路径")
    ap.add_argument(
        "--mode",
        choices=["fixed", "vlm"],
        default="fixed",
        help="fixed=固定四步顺序（默认），vlm=VLM function calling 动态调度",
    )
    ap.add_argument("--start-step", type=int, default=1, metavar="N",
                    help="从第 N 步开始（fixed 模式有效）")
    ap.add_argument("--only-step", type=int, default=None, metavar="N",
                    help="只执行第 N 步（fixed 模式有效）")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    if args.mode == "vlm":
        orchestrator = VLMDrivenOrchestrator(cfg)
        ok = orchestrator.run()
    else:
        orchestrator = TableClearOrchestrator(cfg)
        ok = orchestrator.run(start_step=args.start_step, only_step=args.only_step)

    sys.exit(0 if ok else 1)

