## 🤖 Galbot-1 清桌任务编排器

### 部署环境

**1. 创建 conda 环境**

```bash
conda create -n roboos python=3.10 -y
conda activate roboos
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

# 指定配置文件
python task_orchestrator.py --config config.yaml
```

#### VLM Function Calling 模式（动态调度）

每轮拍当前相机图 + 三张示例图（分别对应 Step 1/2/4 完成状态），由 VLM 通过 function call 决定当前该启动哪个动作、继续等待还是停止。支持 `continue_current_action` / `stop_current_action` 闭环控制，鲁棒性更强但速度略慢。

```bash
# 启动 VLM function calling 模式
python task_orchestrator.py --mode vlm

# 可配合 --config 指定配置文件
python task_orchestrator.py --mode vlm --config config.yaml
```

> 提示词和决策参数在 `vlm_driven_config.yaml` 中配置，判断行为异常时调整 `system_prompt` / `completion_rules` / `decision_rules`。

**两种模式对比**

| | 固定顺序模式 | VLM Function Calling 模式 |
|---|---|---|
| 任务调度 | 代码写死 1→2→3→4 | VLM 每轮决定下一步 |
| VLM 职责 | 只判断当前步骤完成与否（yes/no） | 判断当前状态并决定调用哪个函数 |
| 每次发图数 | 2 张（1 示例 + 当前帧） | 4 张（3 示例 + 当前帧） |
| 速度 | 快 | 略慢 |
| 容错性 | 步骤失败直接终止 | VLM 可重调同一步骤 |
| 推荐场景 | 日常使用 | 需要更强鲁棒性时 |

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

