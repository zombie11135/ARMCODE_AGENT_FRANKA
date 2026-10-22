# 观测模块开发指南

`feat/occlusion-observation-scheduling` 是共享开发分支；`main` 保存稳定版本。
开发更新先进入共享开发分支，经离线回归、相关模型验证和所需实机验收后，
再通过单独的 PR 更新 `main`。CI 不会自动发布到稳定分支。

## 获取代码和提交修改

首次开发克隆完整仓库，以保留观测模块的公共接口和服务依赖：

```bash
git clone --branch feat/occlusion-observation-scheduling https://github.com/zombie11135/ARMCODE_AGENT_FRANKA.git
cd ARMCODE_AGENT_FRANKA
git switch -c feat/observation-your-task
python3.10 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/check_core.py
```

在个人任务分支完成修改和测试后推送，创建目标为
`feat/occlusion-observation-scheduling` 的 PR。网页编辑同样选择开发分支或个人任务分支。
共享开发分支的直接推送与 PR 都运行 `core` CI；以实际检查结果为准。

## 代码地图

| 职责 | 入口 | 开发重点 |
| --- | --- | --- |
| 相机和观测缓存 | `runtime/camera.py`、`runtime/observation_memory.py` | 采集时间、位姿关联、缓存复用和场景变化后的失效 |
| 感知任务调度 | `runtime/perception_pipeline.py` | 有界并发、结果顺序、共享模型资源 |
| Agent 观测动作 | `skill_lib/actions/observation.py`、`observation_cache.py` | 观测调用、结果发布、重新观察 |
| 设备采集 | `hardware/agent_camera.py`、`services/realsense_dual_camera_service.py` | RGB-D、时间戳、内参、相机序列号 |
| 双相机与只读 FK | `occlusion_reasoning/camera_only.py`、`robot_pair.py` | 静止采集组、采集时 FK 和基座系几何 |
| 实例与遮挡关系 | `occlusion_reasoning/observations.py`、`multiview.py`、`relations.py`、`scene.py` | 对应、投影、证据、未决关系 |
| 真实图像分析与融合 | `occlusion_reasoning/real_single_view.py`、`real_fusion.py` | 分割结果、快照完整性、几何对齐、跨视角融合 |
| 固定相机标定 | `hardware/eye_to_hand.py`、`eye_to_hand_capture.py`、`eye_to_hand_ros_worker.py`、`aruco_pose.py` | 外参求解、独立验证、标记位姿检查 |
| 感知模型接口 | `perception_v2/`、`services/` 中的检测、分割与参考识别服务 | 模型输入输出和运行环境 |
| 仿真评估 | `robocasa_occlusion/` | 独立配置仿真环境后的多视角评估 |

已有的 Agent、Gate、抓放和夹爪模块保留在仓库中。观测输出结构或公共接口变动时，
同步检查调用方与对应测试；跨模块改动在 PR 中注明影响。

## 验证

```bash
python scripts/check_core.py
python scripts/prepare_release.py --check
python -m app --help
python -m skill_lab.run_agent_system --help
```

`check_core.py` 显式列出可移植测试，涵盖核心链路、观测缓存、并发管线、参考图库、
模拟相机、双视角几何、标定和硬件接口模拟。无需 API Key、模型权重、相机或 ROS。
测试使用合成数据和模拟设备，不代表真实模型精度或实机效果通过验收。
模型相关的其他测试（如 `test_perception_v2_geometry.py` 会导入 PyTorch）在对应独立环境运行。
不要直接用全仓库 `pytest` 收集实验入口。

提交前只暂存本次修改，再检查实际暂存范围：

```bash
git add <本次修改的文件>
python scripts/prepare_release.py --check-tracked
git diff --cached --stat
git commit -m "Describe observation change"
git push -u origin HEAD
```

## 实机和模型入口

在已配置 RealSense、ROS 和模型依赖的集成电脑使用：

```bash
bash run_occlusion_camera_test.sh capture --help
bash run_occlusion_camera_test.sh capture-base --help
bash run_occlusion_camera_test.sh dual-view --help
bash run_occlusion_camera_test.sh analyze --help
bash run_occlusion_camera_test.sh fuse --help
bash calibrate_overhead_camera.sh --help
bash test_aruco_pose.sh --help
```

脚本读取本机 `config/local.env`。观测使用 `OCCLUSION_CAMERA_PYTHON`，
回退到 `FR3_AGENT_PYTHON` 或当前 `python3`；标定工具使用 `FR3_AGENT_PYTHON`。
实际采集流程和参数见 [双视角设计及实验记录](dual_view_occlusion.txt)、
[固定相机标定说明](eye_to_hand.txt)。文档中的旧绝对路径需替换成本机路径。

静止双视角入口保存每次采集对应的 FK 和几何证据，尚未接入 Agent 自动清障决策。
维护采集组、时间戳、内参、单位和坐标系契约；未知关系和校验失败要保留为未决证据。
历史实验记录中的标定或融合限制仍然适用，以新数据上的独立验证结果为准。

## 数据与配置

`data/object_gallery/` 的清单、参考图片和采集记录已纳入开发分支，特征缓存和运行锁留在本机。
配置使用仓库中的 `*.example.*` 模板。实际设备配置、标定结果、模型、API Key 和 `runs/`
实验产物保持本机管理。几何和缓存算法开发可直接使用现有测试中的合成观测。
