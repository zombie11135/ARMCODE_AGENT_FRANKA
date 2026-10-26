# armcode-agent
机械臂 Agent：从任务规划、感知和 Gate 判断，到技能执行、失败反馈与重规划。
当前主要接入 FR3 / Robotiq，保留显式选择的旧 xArm / ROS 1 链路。

腕部遮挡关系图已接入简短技能链路，可通过 `--enable_geometry_occlusion` 启用
“清障 → 重新观测 → 抓取目标 → 放入容器”的有界循环。
启用参数、缓冲槽位配置和已验证范围见 [Agent 遮挡闭环说明](docs/agent_occlusion_loop.txt)。

## 分支与观测开发

`feat/occlusion-observation-scheduling` 是共享开发分支，`main` 保留稳定版本。
新开发从共享开发分支开始，验证稳定后通过单独的 PR 更新 `main`。
观测模块的代码位置、离线验证和双相机入口见 [观测开发指南](docs/OBSERVATION_DEVELOPMENT.md)。

## 新成员快速开始

这是源码协作基线：包含 Agent、模型适配和硬件接口，模型权重、第三方 SDK、授权、
本机配置和实验采集数据由各机器单独准备；`data/object_gallery/` 参考图库随仓库共享。先在 Python 3.10 环境验证基础链路：

```bash
python3.10 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/check_core.py
python -m app --help
```

以上检查无需 GPU、模型、API Key 或机器人。完整运行需另行配置，详见
[环境与配置](docs/SETUP.md)、[第三方依赖](THIRD_PARTY.md)。
整块数据盘迁移、新电脑依赖和本机配置备份见 [电脑与硬盘迁移说明](docs/machine_migration.txt)。
多人协作使用任务分支和 PR，见 [协作约定](CONTRIBUTING.md)；
首次建立干净 Git 仓库见 [发布指令](docs/RELEASING.md)。

## 从哪里开始阅读

正式入口是 [`app/cli.py`](app/cli.py)，推荐继续使用
`bash run_fr3_agent.sh` 启动；在配置好依赖的 Python 环境中也可运行 `python -m app`。
原有 `python skill_lab/run_agent_system.py` 和 `python -m skill_lab.run_agent_system`
仍然有效，参数保持兼容。`--help` 只显示命令行帮助。

默认 FR3 简短技能链路如下；旧的生成完整执行代码链路仍位于 `agents/agent_system.py`
和 `agents/code_generator.py`，由旧入口选项使用。

```text
app/cli.py                             参数、交互命令、任务入口
  └─ agents/simple_agent_system.py     规划 → 动作生成 → 执行 → 失败重规划
       ├─ planning_agent.py / plan_graph.py
       ├─ simple_code_generator.py    生成 core.actions.SimpleAction 序列
       └─ simple_execution_runtime.py 执行顺序、任务状态、失败报告
            └─ skill_lib/simple_actions.py
                 ├─ skill_lib/actions/  观测、抓取、放置、Gate 与恢复的具体实现
                 ├─ nesyro_gate/      动作前的判断与建议
                 ├─ skill_lib/        观测、抓取、放置与效果验证
                 └─ services/ + hardware/  模型服务和设备适配

runtime/services.py  为规划和执行装配共享服务，通过 core.context.RuntimeContext 注入
runtime/camera.py    管理单次任务的相机缓存；采集与位姿绑定由 hardware/agent_camera.py 负责
```

## 模块职责与协作边界

| 目录 | 负责什么 | 修改时从哪里入手 |
| --- | --- | --- |
| `app/` | 命令行参数、交互命令、任务启动 | 新参数、交互操作和入口配置转换 |
| `agents/` | 规划、任务图、动作生成、执行调度与重规划 | 调整任务步骤、执行顺序、恢复逻辑 |
| `core/` | 公共动作类型、技能接口、返回值、运行上下文 | 修改跨模块接口时协调调用方 |
| `runtime/` | Agent 配置、任务相机缓存、公共服务装配 | 为主链路选择或接入服务，统一两处调用方 |
| `services/` | 检测、分割、OCR、识别、抓取推理等服务 | 在具体服务内实现模型调用，再接入装配层 |
| `perception_v2/` | 离线 RGB-D 感知流水线 | 保存图像上的识别、掩码和点云处理 |
| `nesyro_gate/` | Gate 接口、规则、可学习 Gate 接入与证据 | 调整决策逻辑和输入输出约定 |
| `gate_learning/`、`gate_sim/` | 多模态学习与仿真 | 训练、评价和仿真环境 |
| `skill_lib/` | 技能、组合技能与简短动作门面 | 单项观测、抓放、遮挡处理和效果验证 |
| `anyplace_real/` | 放置相关离线准备、几何与推理工具 | 依照该目录 README 使用独立入口 |
| `hardware/` | FR3、夹爪、相机几何、坐标变换及设备工具 | 驱动、设备生命周期、运动与几何适配 |
| `config/` | 硬件、标定和感知配置 | 与实际设备匹配的配置文件 |
| `skill_lab/`、`tests/` | 实验脚本、兼容入口和回归检查 | 模块开发时补充相应验证 |
| `runs/`、`logs/`、`data/` | 运行产物、日志和参考图库 | `data/object_gallery/` 的清单、图片和采集记录随仓库同步，其余数据和缓存保留本机 |

`skill_lib/combound/` 沿用原有拼写，避免已有导入路径失效。目录职责描述的是当前分工，
还不是强制依赖检查；部分旧技能与硬件适配之间仍有直接调用。

### 配置和公共接口放在哪里

- [`runtime/config.py`](runtime/config.py)：`SimpleAgentSystemConfig`，任务、规划与执行配置。
- [`skill_lib/config.py`](skill_lib/config.py)：`SimpleRobotConfig`，技能执行参数。
- [`hardware/config.py`](hardware/config.py)：硬件 YAML 的读取与设备配置。
- [`core/actions.py`](core/actions.py)：`SimpleAction` 和 `observe / grasp / place` 动作构造函数。
- [`core/context.py`](core/context.py)、[`core/skill_base.py`](core/skill_base.py)：服务上下文、技能接口和 `SkillResult`。
- [`runtime/services.py`](runtime/services.py)：规划上下文、相机和执行上下文的共同装配入口。

历史导入 `agents.simple_agent_system.SimpleAgentSystemConfig`、
`skill_lib.simple_actions.SimpleRobotConfig` 和动作构造函数继续可用；新代码优先从定义所在模块导入。
这一轮提取保留了配置字段、顺序和默认值，配置覆盖行为仍由 CLI 与硬件 profile 处理。

### 多人修改时的约定

1. 单个模块的实现放在对应目录；新增服务通过 `runtime/services.py` 接入主链路。
   `hardware/runtime.py` 继续负责设备装配和旧接口适配。
2. 公共动作、技能返回结构、坐标系或单位变化，应在同一次评审中列出受影响的调用方。
3. 配置提取、文件迁移与算法行为变化分开提交，方便评审和定位问题。
4. 测试和样例随模块开发补充。检查范围按依赖选择，不能把所有实验脚本当作离线测试运行。
5. 实机验证记录对应源码、设备配置和标定版本。没有 `--execute_robot` 只代表不执行动作，
   主流程仍可能初始化相机、模型或读取设备状态。

第一轮结构整理提取了入口、配置、动作类型与公共服务装配。
第二轮将 `SimpleRobotActions` 的实现按职责拆入 `skill_lib/actions/`。
CLI 参数分组和更细的数据接口可在后续独立变更中继续整理。

### 简短动作的实现位置

[`skill_lib/simple_actions.py`](skill_lib/simple_actions.py) 保留 `SimpleRobotActions`
构造函数、动作序列执行、共享状态和原有方法签名；具体实现通过显式调用委托到下列模块。
修改一种动作时，优先进入相应实现文件。

| 实现文件 | 负责内容 |
| --- | --- |
| [`actions/observation.py`](skill_lib/actions/observation.py) | 药盒与容器观测、扫描、缓存写入和失效 |
| [`actions/grasp.py`](skill_lib/actions/grasp.py) | 抓取技能参数与调用、抓取后的观测更新 |
| [`actions/placement.py`](skill_lib/actions/placement.py) | 使用缓存计划放置、槽位偏移、释放与抬升顺序 |
| [`actions/motion.py`](skill_lib/actions/motion.py) | 回 home、观察位姿和现有运动适配调用 |
| [`actions/gate.py`](skill_lib/actions/gate.py) | Gate 前置检查、场景与输入构建、观察探测和决策分支 |
| [`actions/occlusion.py`](skill_lib/actions/occlusion.py) | VLM 遮挡证据、清障恢复、主动视角重试 |
| [`actions/gate_logging.py`](skill_lib/actions/gate_logging.py) | Gate trace、决策/结果日志、JSON 序列化 |
| [`actions/targets.py`](skill_lib/actions/targets.py) | 容器名称、颜色分类和放置槽位计算 |

共享状态仍由一个 `SimpleRobotActions` 实例持有：`observations`、`place_plans`、
`place_counts`、探测/恢复计数以及 Gate 缓存。实现函数显式接收这个实例，直接使用同一份状态；
执行运行时恢复缓存或调用方替换方法后，各模块都能看到更新。
跨职责操作仍经由实例方法调用，例如 Gate 使用 `robot.observe(...)` 获取新观测。
新增实现不要复制这些缓存，也不要在模块导入时初始化模型或设备。

这是保持现有行为的职责拆分；各模块仍共享任务状态，尚未改成可独立部署的服务。
`SimpleRobotActions`、动作构造函数、容器解析辅助函数和 `FALLBACK_INTRINSICS` 的历史导入继续有效。
相关离线回归入口为 `tests/test_simple_actions_components.py`、
`tests/test_gate_runtime_integration.py` 和 `tests/test_gate_experience_logging.py`。

## FR3 / Robotiq 接入

主入口默认读取本机 `config/hardware/fr3.yaml`，使用简短技能链路。
发布包提供 [硬件配置模板](config/hardware/fr3.example.yaml)、
[标定模板](config/calibration/fr3_handeye.example.yaml) 和
[参考图库模板](config/perception/references.example.yaml)，具体配置步骤见
[SETUP](docs/SETUP.md)。实际配置不随源码分发，模板标定状态为 pending。
需要旧 xArm / ROS 1 链路时必须显式加 `--legacy-xarm` 并配置相应环境。

FR3 复用另行部署的 Docker / ROS 2 / MoveIt 2 工作区，Robotiq 2F-85 使用
USB/RS-485 Modbus RTU；相机帧、拍照时位姿、工具和抓取坐标系由 `hardware/` 处理。
模型适配支持本地 AnyGrasp 或 GSNet，SAM3、GroundingDINO、OCR、VLM 和 Gate 接口保留。

`python -m hardware.cli config` 检查配置；`python -m app --help` 显示参数。
实际设备上的操作与验收由集成人员完成，参照历史接口记录
[fr3_hardware.txt](docs/fr3_hardware.txt) 和 [fr3_agent.txt](docs/fr3_agent.txt)。
历史记录中的路径、序列号和实测结论对应原实验电脑，不代表新克隆已具备相同环境。
`--execute_robot` 控制执行动作；省略它仍可能初始化相机、模型或读取设备状态。

## Gate 遮挡抓放仿真（simulation only）

阶段一的可恢复 PyBullet/Panda 环境位于 `gate_sim/`，环境标识为
`GateOccludedPickPlace-v0`。它默认使用 DIRECT/CPU，不导入也不调用真机服务。
安装、demo、恢复验证和固定种子比较命令见
[`docs/gate_sim_phase1.md`](docs/gate_sim_phase1.md)。

当前默认场景除原有目标/遮挡盒外，还包含带本地程序化包装贴图的燕麦盒、
饮料罐、杯状容器和水瓶。它们都是有质量、摩擦和碰撞的动态刚体，并进入 RGB-D
感知、快照和扰动评价；不需要额外下载资产。快速渲染 A--D 四类场景：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli render \
  --output-dir runs/gate_sim/textured_gallery \
  --width 1280 --height 720
```

命令会打印每张 RGB/深度图的实际路径，并生成
`runs/gate_sim/textured_gallery/textured_scene_gallery.png`、四张包含完整 Panda 的
`overview_scene_*.png` 及对应 JSON 报告。overview 展示相机不进入 Gate observation。

录制场景 C 中 Panda 将蓝色遮挡物搬到 buffer 的固定全景视频：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli manip-video \
  --output-dir runs/gate_sim/manipulation_video \
  --width 1280 --height 720
```

场景 E 使用带贴图燕麦盒作为任务目标：蓝色方块压在目标上，操作 Probe 先将
方块搬到 cyan buffer，随后 EXECUTE 把燕麦盒搬入紫色目标容器。生成完整任务视频，
并分别生成 `EXECUTE / VISUAL_PROBE / MANIPULATION_PROBE / REPLAN` 演示：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli gate-videos \
  --output-dir runs/gate_sim/gate_action_videos \
  --width 1280 --height 720
```

`REPLAN` 当前按 action mask 如实展示为不可用和无副作用，不会被伪装成运动成功。

## RGB-D + 任务文字多模态 Gate

`gate_learning/` 提供不依赖上游 VLM confidence 的新 Gate 原型：冻结图文编码器、独立
深度 patch 分支、17 维可信状态/有效性、最多四帧历史以及轻量 Transformer 融合头。
旧 Gate 的实现和默认模式保持不变。架构图、验证状态与训练命令见
[`docs/multimodal_gate_phase1.md`](docs/multimodal_gate_phase1.md)。

## 规则版 NeSyRo Gate 原型

`nesyro_gate/` 是一个用于机器人高层任务规划节点执行前判定的规则版 NeSyRo Gate 原型。它面向 InstructFlow 生成的 instruction graph，在执行 `observe`、`grasp`、`place` 节点前输出：

- `EXECUTE`：当前节点可以执行
- `PROBE_REQUIRED`：需要先进行 safe probe / observe
- `LOCAL_REPAIR`：当前节点代码或参数需要局部修复
- `REPLAN`：需要交给 InstructFlow 重新规划
- `ABORT`：预留的异常终止决策

规则模块本身不直接连接机器人、LLM 或 PDDL。它通过下文的 `SimpleRobotActions`
与执行运行时接入实机闭环，具体观察、恢复和运动由调用方执行。

核心规则：

- Verification 用规则模拟任务规格检查，例如技能是否允许、参数是否缺失、目标物体或盘子是否匹配。
- Validation 用规则模拟技能前置条件检查，例如目标是否可见、可达、可抓，盘子是否可见、有空间、可放置。
- CSC 使用 OCR/VLM/detector 分数融合：`CSC = 0.5 * OCR + 0.3 * VLM + 0.2 * detector`。
- LC 使用符号前置条件规则计算。
- `NeSyConf = CSC x LC`。
- 低置信度时优先返回 `PROBE_REQUIRED`；probe 超过次数或遇到结构性问题时返回 `REPLAN`。

运行 demo：

```bash
python -m nesyro_gate.demo
```

运行测试：

```bash
pytest tests/test_gate.py
```

### 从现有机器人缓存构建 SceneMemory

`nesyro_gate.scene_memory_builder` 可以把 `SimpleRobotActions` 中的缓存转换成 gate 使用的 `SceneMemory`：

```python
from nesyro_gate import build_scene_memory_from_simple_robot, build_task_spec_for_pick_place

scene_memory = build_scene_memory_from_simple_robot(
    observations=robot.observations,
    place_plans=robot.place_plans,
    place_counts=robot.place_counts,
    gripper_holding="蒙脱石散",  # 如果当前夹爪里有目标；没有则传 None
)

task_spec = build_task_spec_for_pick_place(
    target_object_name="蒙脱石散",
    target_plate_name="blue tray",
)
```

构建器会优先使用 `FindAndGraspMedicineSkill` 的 `selected_instance`、`selection_result`、`ocr_verification`、`grasp_input`，以及 `PlaceToContainerSkill` 的 `selected_detection`、`center_base_xyz`、`place_pose_mmrad`。当前仍然是保守规则版：遮挡、可达性、可放置空间等字段会根据已有检测结果和规划结果估计，后续可以替换为真实碰撞检测、IK、占用栅格或主动视觉结果。

### 接入实机闭环

`SimpleRobotActions` 和 `SimpleExecutionRuntime` 已接入 NeSyRo Gate。执行 `observe / grasp / place` 前会自动：

1. 从当前观测缓存和放置规划缓存构建 `SceneMemory`。
2. 将当前动作转换成 `Step`。
3. 调用 `nesyro_gate(...)`。
4. `EXECUTE` 时继续执行真实 skill。
5. `PROBE_REQUIRED` 时调用现有 `observe(...)` 做安全观察后重试 gate。
6. `LOCAL_REPAIR / REPLAN / ABORT` 时停止当前动作，并把 gate 证据写入失败报告，交给后续重规划。

默认启用。运行 agent 时可调参数：

```bash
python skill_lab/run_agent_system.py --simple_actions --execute_robot \
  --nesyro_gate_threshold 0.7 \
  --nesyro_gate_max_probe 3 \
  --nesyro_gate_trace_path runs/current_plan/gate_trace.jsonl
```

如果需要临时关闭：

```bash
python skill_lab/run_agent_system.py --simple_actions --disable_nesyro_gate
```

每次 gate 判定会写入 JSONL trace，包含 `decision / confidence / csc / lc / failed_conditions / suggested_probe / suggested_constraint / scene_objects`，用于实机调参和复盘。

抓取遮挡现在是硬约束：`SceneMemory` 会优先使用显式 `occlusion_level`，其次使用 mask/bbox 可见比例，并在有 RGB-D 深度和 `numpy/cv2` 可用时检测 bbox 内更靠近相机的前景压盖物。只要抓取目标 `occlusion_level > 0.35`，即使 OCR/VLM 语义置信度很高，Gate 也会返回 `REPLAN` 和 `target_occluded(object_id)`，并把该目标在 `SceneMemory` 中更新为 `graspable=False`。

单独测试遮挡反应：

```bash
python -m nesyro_gate.occlusion_demo
```

该 demo 不连接相机或机械臂，会构造无遮挡、轻微遮挡、显式遮挡、mask 可见面积不足、以及可选的深度前景遮挡案例，专门打印 gate 对 `target_occluded` 的反应和 `SceneMemory` 写回结果。

结合实机观察链路测试 Gate，但不执行抓取：

```bash
python skill_lab/test_nesyro_gate_observe.py --target 布洛芬 --plate "blue tray"
```

测试时可以手动摆放几种情况：无遮挡、药盒边缘被挡、药盒上方压住其他物体、只露出文字等。脚本会调用现有观察链路构建 `SceneMemory`，只运行 `grasp(target)` 的 Gate 判定，并打印 `decision / failed_conditions / occlusion_level / graspable / OCR 分数`。

如果目标完全遮挡导致观察链路没有找到稳定匹配的药盒，脚本不会直接退出；它会把失败观察转换为 `visible=False, graspable=False, occlusion_level=1.0` 的保守 `SceneMemory` 目标，再交给 Gate 输出结构化判定。

启用 VLM 遮挡判别：

```bash
python skill_lab/test_nesyro_gate_observe.py \
  --target 布洛芬 \
  --plate "blue tray" \
  --use_vlm_occlusion \
  --vlm_base_url "$ARM_AGENT_API_BASE_URL" \
  --vlm_api_key "$ARM_AGENT_API_KEY" \
  --vlm_model "$ARM_AGENT_API_MODEL"
```

VLM 判别器会重点判断：目标药盒是否可见、是否只有文字露出、是否有其他物体压在或覆盖在药盒上、当前是否安全抓取。它的结果会写入 `SceneMemory` 的 `occlusion_level` 和 `graspable`，再由 Gate 输出最终 `EXECUTE / REPLAN`。

如果启用了 `--use_vlm_occlusion` 但没有提供 VLM 配置，脚本会直接报错退出，不会退回规则版后继续输出 `EXECUTE`。这是为了避免误以为已经使用 VLM。

测试 Gate 触发后移开遮挡物到固定 buffer：

```bash
python skill_lab/test_nesyro_gate_clear_occlusion.py \
  --target 布洛芬 \
  --plate "blue tray" \
  --vlm_base_url "$ARM_AGENT_API_BASE_URL" \
  --vlm_api_key "$ARM_AGENT_API_KEY" \
  --vlm_model "$ARM_AGENT_API_MODEL"
```

这个脚本默认是 dry-run：它会观察目标、调用 VLM 遮挡判别、运行 Gate；如果 Gate 返回 `REPLAN` 且包含 `target_occluded`，会从当前检测候选中选一个与目标 bbox 重叠的遮挡候选，并只规划遮挡物抓取，不移动机械臂。真实搬移时显式加 `--execute`：

```bash
python skill_lab/test_nesyro_gate_clear_occlusion.py \
  --target 布洛芬 \
  --plate "blue tray" \
  --vlm_base_url "$ARM_AGENT_API_BASE_URL" \
  --vlm_api_key "$ARM_AGENT_API_KEY" \
  --vlm_model "$ARM_AGENT_API_MODEL" \
  --buffer_pose_mmrad 280 -260 170 3.14 0 -1.57 \
  --execute
```

如果自动选择的遮挡候选不稳定，可以手动指定遮挡物 bbox：

```bash
python skill_lab/test_nesyro_gate_clear_occlusion.py \
  --target 布洛芬 \
  --vlm_base_url "$ARM_AGENT_API_BASE_URL" \
  --vlm_api_key "$ARM_AGENT_API_KEY" \
  --vlm_model "$ARM_AGENT_API_MODEL" \
  --blocker_bbox 517 180 760 360
```

当 Gate 输出 `REPLAN / LOCAL_REPAIR / ABORT` 时，观察测试脚本会写出：

```text
runs/current_plan/failure_report.json
```

其中包含 `failure_type=nesyro_gate_replan`、`failed_conditions`、`suggested_constraint` 和 VLM/Gate 证据。完整 `--simple_actions` 闭环会自动把这类 failure report 交给 `ReplanningAgent`；约束分类器会把 `target_occluded(药名)` 转成“禁止直接抓取、先主动观察/移除遮挡”的 InstructFlow 约束。

在完整实机闭环中启用：

```bash
python skill_lab/run_agent_system.py --simple_actions --execute_robot \
  --enable_vlm_occlusion_gate \
  --enable_occlusion_recovery \
  --occlusion_buffer_pose_mmrad 280 -260 170 3.14 0 -1.57 \
  --occlusion_min_depth_valid_ratio 0.35
```

`--enable_vlm_occlusion_gate` 默认复用 `--api_base_url / --api_key / --api_model`，也可以用 `--vlm_occlusion_base_url / --vlm_occlusion_api_key / --vlm_occlusion_model` 单独指定。

`--enable_occlusion_recovery` 默认关闭。启用后，抓取前 Gate 若返回
`target_occluded`，运行时会调用正式的
`ClearOcclusionToBufferSkill`：用检测候选、目标 bbox 重叠关系和深度前后关系定位上层遮挡物，使用 AnyGrasp 规划抓取，将其放到固定 buffer，然后清除旧观测、重新观察目标并再次执行 VLM 和 Gate。默认最多恢复一次。

为了避免 D435i 距离过近导致坏深度被送入 AnyGrasp，清障技能会检查遮挡物 ROI 的有效深度比例。低于 `--occlusion_min_depth_valid_ratio` 时不会移动机械臂，而是保留 `REPLAN`，并返回 `move_camera_farther_and_reobserve` 建议。目标完全不可定位时默认也不会盲目搬动物体；只有显式加入 `--occlusion_allow_unlocalized_target` 才允许探索性清障。

清障定位不使用 VLM 预测 bbox。VLM 只提供遮挡物外观描述，GroundingDINO 用该描述定位候选；覆盖超过整幅图 50% 的大框默认会被过滤。随后系统从 bbox 内提取最近深度层的连通 mask，只有近层点云支撑率达到 `--occlusion_min_foreground_support_ratio 0.15` 才会交给 AnyGrasp，避免 bbox 中残留的下层药盒成为最佳抓取点。

如果当前 D435i 视角中的遮挡物深度为空或近层 mask 不足，清障恢复会默认启动主动视角：依次移动到 `z=550 mm`、`y=0/-120/120 mm` 重新拍摄并重跑 VLM、检测和深度检查。可使用 `--disable_occlusion_active_view` 关闭。每次清障会保存 `runs/current_capture/occlusion_recovery_blocker_selection.png` 和 `occlusion_recovery_foreground_mask.png`，分别用于核对最终遮挡物 bbox 与实际送入 AnyGrasp 的点云区域。
