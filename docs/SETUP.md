# 开发环境与本机配置

## 普通协作者：先跑纯 CPU 检查

使用 Python 3.10，在仓库根目录执行：

```bash
python3.10 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/check_core.py
python -m app --help
```

该环境用于规划、公共接口、动作委托和 Gate 规则回归，不需要模型权重、API Key、
CUDA、机器人或相机。`requirements-dev.txt` 中的 headless OpenCV 仅用于离线处理，
不是运行相机窗口、模型推理或实机的完整环境。不要把不同模块的 requirements 全部装进同一个环境。

`tests/` 还保留了依赖实机配置、外部工作区或特定模型环境的测试；
`skill_lab/` 与根目录 `test_xarm.py` 是实验入口。不要使用全仓库 `pytest` 作为通用验收命令。
已验证可移植的测试由 `scripts/check_core.py` 显式列出，CI 使用同一个入口。

## 集成电脑：按需配置

在独立的实机环境中安装和准备相应依赖，参照 [第三方依赖说明](../THIRD_PARTY.md)。
以下命令只在**新克隆目录**中执行；已有配置的电脑不要覆盖原文件：

```bash
cp config/local.env.example config/local.env
cp config/hardware/fr3.example.yaml config/hardware/fr3.yaml
cp config/perception/references.example.yaml config/perception/references.yaml
cp config/calibration/fr3_handeye.example.yaml config/calibration/fr3_handeye.yaml
```

填写硬件 YAML 的 ROS 工作区、相机序列号、夹爪端口、模型路径和推理解释器。
模板的 `stack: fake` 仍需要外部 ROS / MoveIt 工作区；它不是内置仿真器。
真实设备需使用与其匹配的 profile 和经过验证的标定。
标定模板是 `status: pending`、空矩阵，不能用于抓取；不要用单位矩阵冒充测量结果。
这些本机文件均被 Git 忽略，发布包不包含旧电脑的实际标定和图库。

FR3 适配器依赖另行部署的 Docker Compose / ROS 2 / MoveIt 2 工作区，至少包括
`compose.yaml`、`real` / `fake` 服务和工作区中的机器人配置。该工作区不随 Python 源码分发。
本仓库 `hardware/fr3.py` 和 `hardware/fr3_worker.py` 描述了实际接口约定。

在 `config/local.env` 配置解释器和 API，使用 shell 的 `export NAME='value'` 格式。
该文件由 `run_fr3_agent.sh` 等启动脚本加载；直接执行 Python 时先手动
`source config/local.env`。文件是本机可信 shell 配置，不要从陌生来源复制执行。

- `FR3_AGENT_PYTHON`：Agent 环境中的 Python。
- `ARM_AGENT_PYTHON`：图库采集环境中的 Python。
- `GSNET_PICK_PYTHON` / `ANYGRASP_PYTHON`：对应独立环境中的 Python。
- `ANYGRASP_COMPAT_LIBS`：如 SDK 需要，配置其兼容库目录。
- `ARM_AGENT_API_BASE_URL`、`ARM_AGENT_API_KEY`、`ARM_AGENT_API_MODEL`：所选服务的地址、凭据和模型名。

API 地址和 Key 的源码默认值为空；基础离线检查不需要它们。依然支持已有的
`PLANNER_*` 环境变量及 CLI 参数，具体优先级以 `python -m app --help` 和 `app/cli.py` 为准。
使用远程 LLM/VLM 时由集成人员填写自己的服务配置。

旧 xArm 实验脚本和部分历史配置类仍保留旧路径默认值，必须显式配置后再使用。
主 FR3 入口以本机 YAML 为准；本次发布没有重写旧实验脚本的执行行为。
`docs/*.txt`、感知和 AnyPlace 的旧 README 中的绝对路径属于原实验环境记录，
不能直接作为新电脑的安装命令；旧实验的照片、manifest 和标定也不在发布包内。

完成本机配置后，先使用 `python -m hardware.cli config` 检查配置。
实机流程、模型验证、标定复核和最终运动验收仍由集成人员在对应设备上完成。
没有 `--execute_robot` 仅意味着不执行动作，主任务仍可能初始化模型、相机或读取设备状态。
