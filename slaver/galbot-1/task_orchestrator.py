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
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests
import yaml


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
        cam = cfg["camera_server"]
        self.snapshot_url = "http://{}:{}{}".format(
            cam["host"], cam["port"], cam["snapshot_path"]
        )
        self.api_key = vlm["api_key"]
        self.api_base = vlm["api_base"]
        self.model = vlm["model"]
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
        import requests, base64
        try:
            r = requests.get(self.snapshot_url, timeout=5)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        except Exception as e:
            self.log.warning("[vlm] 抓帧失败: %s", e)
            return None

    def _ask(self, b64, prompt, step_key=None):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
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
                {"type": "text", "text": prompt},
            ]
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=16,
                temperature=0.0,
                extra_body={"enable_thinking": False},
            )
            answer = resp.choices[0].message.content.strip().lower()
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

from dataclasses import dataclass
from typing import Callable


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
        try:
            self.client.post(api_path)
        except Exception as e:
            return StepResult(step_id, name, False, "启动失败: " + str(e), time.time() - t0)
        if pre_sleep > 0:
            self.log.info("[%s] 等待初始化 %.1fs ...", name, pre_sleep)
            time.sleep(pre_sleep)
        done, polls = self.vlm.poll_until_done(name)
        self.client.stop()
        return StepResult(step_id, name, done,
                          "VLM polls={}, done={}".format(polls, done),
                          time.time() - t0)

    def _run_pick_bag(self):
        return self._infer_step(1, "pick_bag", "/api/pick_bag", pre_sleep=3.0)

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
        return StepResult(3, name, ok, msg, time.time() - t0)

    def _run_sweep_trash(self):
        return self._infer_step(4, "sweep_trash", "/api/sweep_trash")

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
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Galbot-1 清理桌面长程任务编排器")
    ap.add_argument("--config", default=_DEFAULT_CONFIG, help="配置文件路径")
    ap.add_argument("--start-step", type=int, default=1, metavar="N",
                    help="从第 N 步开始（断点续跑，默认 1）")
    ap.add_argument("--only-step", type=int, default=None, metavar="N",
                    help="只执行第 N 步（调试用）")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    orchestrator = TableClearOrchestrator(cfg)
    ok = orchestrator.run(start_step=args.start_step, only_step=args.only_step)
    sys.exit(0 if ok else 1)
