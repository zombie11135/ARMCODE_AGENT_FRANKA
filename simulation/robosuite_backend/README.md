# Panda 桌面仿真 Backend（第一阶段）

独立的 robosuite + MuJoCo 技能适配层。提供 Panda、桌面、红蓝两个动态方块与两个
命名放置区，通过 OSC_POSE 控制器实际夹取、抬升、搬移、释放。技能执行不直接
设置物体位置，不使用吸附、焊接或瞬移；仅环境初始化和 reset 设置初始 qpos。

当前是固定两方块、无障碍桌面上的脚本化俯抓基线，使用仿真物体真值规划抓取。
目标区是桌面上的可视标记，并非有侧壁的容器；没有接入真实视觉识别、遮挡推理、
运动规划、学习 Memory、PEIP 或物理恢复。**没有接入生产调度器，也没有修改 CaP / Hybrid。**

## 安装

在 SSD 活动仓库运行：

```bash
cd '/home/wpy/projects/armcode-agent (1)/armcode-agent'
SIM_PYTHON=/home/wpy/armcode-runtime/py311/bin/python3.11 \
  bash scripts/setup_robosuite_backend.sh
```

其他机器使用可用的 Python 3.11：

```bash
SIM_PYTHON=python3.11 bash scripts/setup_robosuite_backend.sh
```

只向项目的 `.venv-robosuite/` 安装，不激活、升级或修补现有 FR3 / RoboCasa 环境。
正常安装需要访问 PyPI 和 GitHub；不安装系统驱动、不启动模型服务、不访问机器人。
使用 [robosuite 官方安装方式](https://robosuite.ai/docs/installation.html) 和
[OSC_POSE 控制器](https://robosuite.ai/docs/modules/controllers.html)，版本固定如下：

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.11（本机验证 3.11.15） |
| robosuite | 1.5.2，提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| MuJoCo | 3.3.1 |
| NumPy / SciPy / Numba | 2.2.5 / 1.15.3 / 0.61.2 |

其余依赖见根目录 `requirements-robosuite-backend.txt`。完整运行版本写入每次实验
的 `manifest.json`。本机终端网络受限，已通过以下离线方式建立并验证独立环境：

```bash
SIM_PYTHON=/home/wpy/armcode-runtime/py311/bin/python3.11 \
SIM_OFFLINE_SITE="$PWD/.venv-robocasa/lib/python3.11/site-packages" \
SIM_ROBOSUITE_SOURCE="$PWD/dependencies/robosuite" \
  bash scripts/setup_robosuite_backend.sh
```

离线安装逐一核对版本及 robosuite 提交，复制所需包和源码，不依赖原环境可启动，
不复制其 editable 路径配置，不修改提供包的环境。它需要同平台的 CPython 3.11
包副本，不能用作跨平台安装。在线下载路径在本次受限终端中未验证。

## 运行与复现

两物体示例：

```bash
.venv-robosuite/bin/python -m simulation.robosuite_backend.demo --seed 7
```

自动创建新的 `runs/robosuite/<时间戳>/`。如指定 `--output-dir`，必须使用不存在的目录，
以免覆盖已有实验。可通过 `--config config/simulation/robosuite_panda.json` 选择独立配置。
示例调用原有 `runtime.cap_policy.PolicyInterpreter`，只注入两个技能布尔闭包；
策略文本固定，完全不调用 LLM 服务。它用于验证受限接口绑定，不代表完整 Hybrid 集成。

全景画面与抓放过程动画：

```bash
.venv-robosuite/bin/python -m simulation.robosuite_backend.visualize --seed 7
```

保存能看到完整 Panda 的 960×720 初末 RGB-D，以及实际物理执行过程的
`pick_place.gif`。动画约为仿真时间的 3 倍速，首尾画面额外停留；不代表真机速度。
此入口只在独立渲染脚本中调整相机和记录帧，不修改 Backend 默认视角或控制逻辑。

两个**全新进程**执行相同任务，比较物理事件与初末 RGB-D：

```bash
.venv-robosuite/bin/python -m simulation.robosuite_backend.smoke --seed 7
```

smoke 在每个进程中执行两次抓放，检查抬升及任务完成，再比较剔除墙钟字段后的完整
事件/结果，并以 `atol=1e-7, rtol=0` 比较 RGB-D 和相机矩阵；失败返回非零退出码。
相同软件、CPU/渲染条件下可复现。跨 GPU、驱动、MuJoCo 版本不保证逐值相同。
`reset()` 从配置种子重建环境与控制器；`reset(seed=8)` 换种子。扰动使用独立派生 RNG。

针对模拟器的最小测试集：

```bash
.venv-robosuite/bin/python -m pytest -q tests/test_robosuite_backend.py
```

测试包括 RGB-D 几何投影与颜色方向、观测不推进物理时间、相同/不同种子、两物体
接触抓放、错误名称、执行超时、失败后停止、错误完成条件、策略顺序约束、扰动入口、
独立日志与防覆盖。没有仿真依赖的环境会跳过该文件；正式验收应在独立环境中运行，
不能把 skip 当作通过。

## 接口与数据约定

```python
from simulation.robosuite_backend import RobosuiteBackend, SimConfig

with RobosuiteBackend(SimConfig(seed=7), log_path="runs/my_sim/events.jsonl") as sim:
    observation = sim.observe("red_cube")
    result = sim.pick_place("red_cube", "red_goal")
    status = sim.task_status()
```

`observe(target=None)` 和 `pick_place(target, container)` 返回原有 `core.skill_base.SkillResult`，
不扩展公共类型。未知名称返回 `success=False`，不推进物理。抓放失败返回失败阶段与
证据，并锁定后续搬运直至显式 reset；不会自动重试或恢复。底层异常保留日志后上抛。
Backend 为单线程同步对象，所有仿真读写及图像采集由同一线程持有。

| 数据 | 约定 |
| --- | --- |
| `data.robot` | 7 轴关节角/角速度、两指位置、末端位姿、阶段及 halted 状态 |
| `data.objects` | 两物体位姿及速度，`source=simulator_ground_truth`，可包含被遮挡物体 |
| `data.camera.rgb` | `uint8[H,W,3]`，RGB，图像左上角原点 |
| `data.camera.depth_m` | `float32[H,W]`，已从深度缓冲转换为光轴 Z 深度，单位米 |
| `K` | 对应输出尺寸的相机内参 |
| `T_world_from_camera_optical` | 光学坐标到仿真世界；光学 X 右、Y 下、Z 前 |
| 位姿 | `frame=sim_world`，米、四元数 XYZW；世界 Z 向上，台面 Z=0.8 m |
| `sim_time_s` / `step` / `episode` | 仿真物理时间、控制步及 reset 代号，可用于证据版本绑定 |
| 抓放结果 | `lifted`、实际抬升高度、阶段、放置判据、最终位姿、控制步数及两种时长 |

物体真值是阶段一的明确能力，**observe 成功仅表示取得状态，不证明视觉可见或身份识别成功**。
不要将这里的位置、Panda 默认工具或世界变换用于 FR3。没有读取任何历史硬件标定或坐标。
在 Python 接口中图像为 NumPy 数组；JSONL 不嵌入图像，示例将 RGB-D 单独保存为 NPZ/PNG。

完成检测按实测状态计算：目标中心 XY 误差 < 3.5 cm、中心高度误差 < 8 mm、接触台面、
与夹爪无接触、线速度 < 1.5 cm/s、角速度 < 0.15 rad/s，连续满足至少 12 个控制步。
任务完成要求两个物体分别位于指定目标区。抓放成功还必须通过双指接触和抬升 > 6 cm。

## 输出与时间

每次示例保存：

- `config.json`、`policy.py`、`manifest.json`：参数、实际策略、版本和实际渲染器。
- `before/after_rgb.png`、`before/after_rgbd.npz`、`before/after_state.json`：初末观测与状态。
- `events.jsonl`：独立追加式事件，含 reset、观测、控制阶段、技能结果及扰动事件。
- `summary.json`：技能执行结果和最终完成判据。

`sim_duration_s` 是步进的物理时间，`wall_duration_s` 是当前主机计算耗时。
二者都不表示真实机械臂执行耗时；不导入生产 `runtime.timing`，不生成生产任务计时报表。
本机验证使用 EGL + Mesa llvmpipe 软件渲染，具体设备号以 manifest 为准。
首次执行可能包含 Numba 编译；它不改变物理时间。

默认无窗口，启动时探测可用 EGL 设备，也可在运行前显式设置：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  .venv-robosuite/bin/python -m simulation.robosuite_backend.demo
```

没有 EGL 时，可在系统已具备 OSMesa 的前提下使用 `MUJOCO_GL=osmesa`；该备用渲染路径
未在本机验证。缺少 mink、robosuite_models、GymWrapper 的上游提示不影响本任务；
本模块只使用原生 Panda、OSC_POSE 和 robosuite 环境，不需要安装这些可选组件。

## 后续扰动与 Runtime 集成

`disturbance_hook(env, *, step, sim_time_s, rng)` 在每个控制步之前调用，包括 reset 的
稳定步；默认 `None`。它可在未来通过 `env.sim` 施加力，并返回可 JSON 序列化的事件
详情，由 Backend 记入独立日志。瞬时外力的清除由 hook 负责。调用者需要在 reset 时
保持自身状态可复现；推荐按 step 调度，不保存跨 episode 的可变计数。
此入口不暴露给生成策略，不包含预置物理扰动、恢复策略或 PEIP。

正式 Runtime 集成的待决接口与文件位置见 [INTEGRATION.md](INTEGRATION.md)。

## 本次验收记录

2026-11-09，本机独立环境完成：

- `tests/test_robosuite_backend.py`：9 passed。
- seed=7，两个全新进程的两物体抓放 smoke：通过；物理事件/结果一致，RGB-D 比较通过。
- 现有 FR3 Python 环境中运行 CaP / Hybrid 的 mock 单元测试：38 passed，未执行硬件动作。
- 独立环境 `pip check`：通过。对开工时 483 个既有受版本控制及未跟踪文件做内容哈希核对，均未改变。

本次实验产物位于项目 `runs/robosuite/phase1_smoke_seed7/`，其中
`smoke_report.json` 为复现结论，`run_a/`、`run_b/` 保存各自的日志、观测、manifest
与 summary。该目录属于忽略的运行产物；复跑请用默认新目录或新的 `--output-dir`。
