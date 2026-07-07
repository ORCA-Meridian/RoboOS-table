## 🤖 Galbot-1 清桌任务编排器

### 部署环境

**1. 创建 conda 环境**

```bash
conda create -n roboos2 python=3.10-y
conda activate roboos2
```

**2. 克隆代码**

```bash
git clone https://github.com/FlagOpen/RoboOS.git
cd RoboOS

```

**3. 安装 RoboOS 依赖**

```bash
pip install requests pyyaml openai  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**4. 安装 FlagScale**

```bash
git clone https://github.com/FlagOpen/FlagScale
cd FlagScale
pip install -e .
# Windows 下如果 pip install -e . 报错，改用：
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation
cd ..
```

**5. 安装编排器额外依赖**

```bash
pip install openai
```

---


> 文件路径：`RoboOS/slaver/galbot-1/task_orchestrator.py`

### 任务流程

| 步骤 | 名称 | 方式 | 说明 |
|---|------|------|------|
| Step 1 | `pick_bag` | VLM 推理 | 从地面拾起垃圾袋放上桌右侧 |
| Step 2 | `bag_largeitems` | VLM 推理 | 将桌面大件垃圾装袋（龙虾饭盒除外） |
| Step 3 | `replay_towel` | Replay 轨迹 | 播放拿毛巾轨迹，毛巾放上桌 |
| Step 4 | `sweep_trash` | VLM 推理 | 用毛巾将龙虾擦进饭盒，装入垃圾袋 |

---


**目录结构**

```
slaver/galbot-1/
├── task_orchestrator.py      # 主程序
├── config.yaml               # 机器人 / VLM / 相机配置
├── vlm_driven_config.yaml    # VLM function calling 模式专用提示词配置
└── img/
    ├── pick_bag.png           # Step 1 完成示例图
    ├── bag_large_items.png    # Step 2 完成示例图
    └── sweep_trash.png        # Step 4 完成示例图
```

**前置服务**（需在机器人端先启动）

```bash
# 机器人 HTTP 服务
python robot_serverclear_table.py

# 相机服务
python camera_viewer.py
```

---

### 启动方式

#### 固定顺序模式（默认，稳定）

按 Step 1 →2 → 3 → 4 固定顺序执行，每步通过 VLM yes/no 判断完成后进入下一步。

```bash
cd slaver/galbot-1

# 完整运行四步
python task_orchestrator.py

# 从第 N 步开始（断点续跑）
python task_orchestrator.py --start-step 2

# 只跑第 N 步（调试单步）
python task_orchestrator.py --only-step 3

# 指定配置文件
python task_orchestrator.py --config config.yaml
```

#### VLM Function Calling 模式（动态调度）

每轮拍当前相机图 + 三张示例图，由 VLM 通过 function call 决定下一步调用哪个函数。支持 `continue_current_action` / `stop_current_action` 闭环控制。

```bash
python task_orchestrator.py --mode vlm
```

---

### 每次启动前需要修改的配置

#### `config.yaml`（固定模式和 VLM 模式都需要）

```yaml
galbot_server:
  host: "172.16.20.48"      # ← 机器人实际 IP

camera_server:
  host: "172.16.20.48"      # ← 机器人实际 IP

vlm:
  api_key: "sk-xxx"         # ← 支持视觉输入的 API Key
  api_base: "https://..."   # ← 模型服务端点
  model: "qwen3.7-plus"     # ← 支持图像输入的模型名
  post_done_sleep:
    pick_bag: 10.0          # ← 固定模式：VLM 判断完成后等待机器人降回高度的秒数

replay:
  parquet_path: "/home/galbot/vla_client/拿毛巾.parquet"  # ← 毛巾轨迹文件路径
```

#### `vlm_driven_config.yaml`（仅 `--mode vlm` 需要）

```yaml
vlm_driven:
  max_rounds: 200           # ← 最大决策轮数
  decision_timeout: 60      # ← 单次 VLM 请求超时秒数
  api_retries: 2            # ← 接口失败重试次数
  stop_delays:
    run_pick_bag: 10.0      # ← VLM 调用 stop 后等待秒数（让机器人手臂降回正常高度）
```

提示词（`system_prompt` / `completion_rules` / `decision_rules`）一般不需要每次修改，只在 VLM 判断行为异常时调整。

---

### 常见参数速查

| 参数 | 文件 | 说明 |
|------|------|
| `galbot_server.host` | `config.yaml` | 机器人 IP |
| `vlm.api_key` | `config.yaml` | VLM API Key |
| `vlm.model` | `config.yaml` | 使用的视觉模型名 |
| `vlm.post_done_sleep.pick_bag` | `config.yaml` | 固定模式 pick_bag 完成后延迟停止秒数 |
| `vlm_driven.stop_delays.run_pick_bag` | `vlm_driven_config.yaml` | VLM 模式 stop 后延迟停止秒数 |
| `vlm_driven.max_rounds` | `vlm_driven_config.yaml` | VLM 模式最大决策轮数 |
| `replay.parquet_path` | `config.yaml` | 毛巾轨迹 parquet 文件路径 |
```