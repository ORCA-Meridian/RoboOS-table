import base64
import logging

import requests


class CameraClient:
    """Fetches JPEG snapshots from the robot camera service."""

    def __init__(self, cfg: dict, log: logging.Logger):
        self.snapshot_url = "http://{}:{}{}".format(
            cfg["host"], cfg["port"], cfg["snapshot_path"]
        )
        self.log = log

    def snapshot_b64(self, log_prefix: str) -> str | None:
        try:
            response = requests.get(self.snapshot_url, timeout=5)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception as exc:
            self.log.warning("[%s] 抓帧失败: %s", log_prefix, exc)
            return None
