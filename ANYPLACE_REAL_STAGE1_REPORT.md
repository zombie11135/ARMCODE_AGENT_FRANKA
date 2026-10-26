# AnyPlace 真机复现 Stage 1 报告

日期：2026-09-10（Asia/Shanghai）<br>
项目：`/home/wpy/pythonproject/armcode-agent`<br>
项目提交：`ab7a00b842ba5ac3d2c7fcb0a50eb7f2c48fbfb7`<br>
工作分支：`phase1-learnable-gate`（开始时已有未提交的 Gate 仿真改动，本轮未覆盖）

## 结论

已建立 AnyPlace 的最小离线入口：可将一组保存的真实、对齐 RGB-D 转为目标点云、承载物点云和人工确认的局部区域，调用官方 AnyPlace 推理函数，导出放置变换、变换后点云、叠加图、坐标检查及不可执行的离线计划。官方样例和真实数据都完成了实际模型推理。

当前 **不能进入首个真机放置**。本轮没有向机械臂或夹爪下发任何命令。最终真实候选在简化承载面检查中估计穿透约 `0.0292 m`，同时项目实际硬件接口与任务给出的 Franka Panda + Robotiq 2F-85 不一致，实际 TCP、夹持物相对 TCP 的变换和标定溯源也不完整。`offline_plan.json` 因此固定为 `execution_allowed=false`，没有机器人 waypoint 或夹爪命令。

## 审计结果：严格区分证据状态

### 已实测（本轮实际执行）

- 项目目录和 Git：当前项目可运行；开始时已有 `.gitignore`、`README.md`、`gate_sim/` 等未提交改动，均保留。本轮使用独立 `anyplace_real/` 子目录，仅额外修正一个共用相机服务文件。
- 机器人只读状态：连接 `192.168.1.237` 成功，SDK `1.17.3`，控制器固件 `v2.2.2`，设备为 6 轴 xArm（序列号 `XI1304`），`get_position` 返回 `[350.000061, -0.000035, 449.999969, 3.14, -0.0, -1.570001]`（mm/rad），状态 `2`，错误/告警 `[0, 0]`。连接后只调用读取接口并断开，未调用 `motion_enable`、`set_state`、`set_position` 或故障复位。
- 控制器 TCP/world offset 只读值均为六维全零。它不能证明夹爪实际工具中心位于法兰原点，反而说明控制器内没有可复用的非零工具补偿。
- RealSense 枚举：只检测到 `Intel RealSense D455`，序列号 `213522251050`。配置中的腕部相机 `243222070053` 未检测到，因此没有现场补拍。
- ROS1 `/grip` 查询未能连接 ROS master；没有调用该服务，也没有操作夹爪。
- AnyPlace 官方样例 `lid_0_on_cup_0` 实际推理成功，模型加载、预处理和输出结构均正常；返回 5 个 `4x4` 变换。
- 保存的真实 RGB-D 实际推理成功；返回 5 个 `4x4` 变换，导出了点云和叠加图。
- 同 seed、同输入、同参数重复真实推理，选中变换最大绝对差为 `0.0`（容差 `1e-7` 通过）。
- 轻量测试：`2 passed`；所有新增 Python 文件及相机服务通过编译检查，`git diff --check` 通过。

### 仅发现代码或历史记录，未在本轮硬件实测

- 相机接口：[services/realsense_dual_camera_service.py](services/realsense_dual_camera_service.py) 使用 `rs.align(rs.stream.color)`，从同一个 aligned frameset 读取 RGB 与深度；[skill_lib/perception/capture.py](skill_lib/perception/capture.py) 对外返回图像路径、内参、时间戳和深度尺度。
- 配置中所谓腕部相机序列号为 `243222070053`，历史说明是 D435i；本轮设备枚举没有看到它，不能写成“当前 D435i 可用”。
- 保存快照 `runs/grasp_snapshots/20260520_131624_西瓜霜/grasp_input.json` 记录：RGB/深度均为 `1280x720`，对齐内参 `fx=909.59, fy=909.8, cx=658.97, cy=355.42`，旧式 `depth_scale=1000.0`（除数语义），拍摄时 EE 位姿 `[0.350000031, -0.000000011, 0.449999939, 3.14, 0.0, -1.570001]`（m/rad）。这是历史文件记录，不是本轮同步读取。
- 手眼文件 [config/calibration/wrist_handeye.yaml](config/calibration/wrist_handeye.yaml) 声明 `camera_to_ee`、单位米。数值旋转行列式 `1.000000006`、正交 Frobenius 误差 `1.15e-8`；但文件没有标定日期、残差、相机序列号、机器人序列号或工具配置绑定。
- 机器人运动接口是 xArm SDK 的 `get_position` / `set_position`，现有抓取规划在 [skill_lib/manipulation/plan_grasp_with_anygrasp.py](skill_lib/manipulation/plan_grasp_with_anygrasp.py) 中加入默认 `gripper_length=0.12 m` 的经验补偿。该值没有被当作实测 TCP 使用。
- 夹爪实现指向 `/home/wpy/gt200_ws` 的自定义 GT200/飞特舵机 ROS1 `/grip` 服务（0 关、1 开），不是 Robotiq 2F-85 驱动。
- 项目里存在 AnyGrasp 抓取候选和直接 xArm 笛卡尔运动代码，但没有发现可直接复用且与 Franka + Robotiq + 当前碰撞场景绑定的放置运动规划入口。

### 仍缺失

- 当前项目中的 Franka Panda 状态/规划接口和 Robotiq 2F-85 状态/控制接口；代码搜索没有找到这两套真机实现。
- 在线腕部 D435i，以及一组新鲜的、带设备序列号、硬件时间戳、RGB/深度帧号与拍摄前后机器人位姿包络的同步采集。
- 经验证的实际 TCP、法兰到夹爪 TCP、夹持后 `T_TCP_object`，以及与这些工具参数绑定的手眼标定质量报告。
- 完整目标几何或足够覆盖的多视角点云。当前目标点云是单视角可见表面，厚度方向信息不足。
- 候选的网格碰撞、承载接触、稳定性、可达性、IK 和轨迹碰撞验证。

## 实际复用与修复

### 相机与深度

修复了 [services/realsense_dual_camera_service.py](services/realsense_dual_camera_service.py) 中的一个几何风险：旧实现先保存原始 depth stream 内参，随后把深度对齐到 color，却仍返回原始深度内参。现在对齐后从 `depth_frame.profile` 读取内参，并增加以下元数据：

- 明确的米乘数 `depth_unit_m_per_raw_unit`；
- `depth_aligned_to_color`；
- 设备名称/序列号；
- RGB 与 depth 各自帧号和传感器时间戳；
- RGB-depth 传感器时间差。

该修复保持原有 `depth_scale` 字段兼容。由于腕部相机不在线，本轮只完成代码级和静态验证，没有把此修复标为腕部硬件实测。

保存真实深度为 `uint16`，本轮显式使用 `0.001 m/raw-unit`，不再依赖含义模糊的 `depth_scale`。目标有效深度约 `0.332–0.337 m`，承载桌面裁剪约 `0.369–0.378 m`。

### AnyPlace 代码、环境与权重

- 官方代码：`/home/wpy/anyplace-main`，来自 `/home/wpy/下载/anyplace-main.zip`。
- 版本证据：zip comment 为 `3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e`；本轮用 `git ls-remote https://github.com/ac-rad/anyplace.git` 核对，远端 `main/HEAD` 同为该提交。解压目录自身没有 `.git`。
- 官方仓库：<https://github.com/ac-rad/anyplace>
- 环境：`/home/wpy/.conda/envs/anyplace`，Python `3.8.20`，PyTorch `1.13.1+cu117`，torchvision `0.14.1+cu117`，NumPy `1.22.4`，Open3D `0.19.0`，torch-scatter `2.1.1+pt113cu117`，torch-cluster `1.6.1+pt113cu117`，knn-cuda `0.2`，Ninja `1.13.0`。
- 计算资源：RTX 2080 Ti 11264 MiB，驱动 `535.230.02`，PyTorch CUDA `11.7` 可用。
- 没有重新训练、降级包或修改机器人/base 环境。首次 KNN 导入需要把该环境已有的 Ninja 与 CUDA headers 加入 `PATH/CPATH`，之后 JIT 扩展加载成功。

权重 SHA-256：

| 权重 | SHA-256 |
|---|---|
| `anyplace_multitask/model.pth` | `d3d33f0a279633c25f252960a208d4b4447a756f0cff8e94be0faadc20dc5be5` |
| `hanging/model.pth` | `a6ad794bb182db9ba48abddfbd62787a034ce53730e9edf1e48c5ab150663248` |
| `peg_insertion/model.pth` | `8a04b504a0914c5cdebc413df5755ea06f6461d6b5a5d7c27881255d5f857103` |
| `stacking/model.pth` | `3517eef1d2699a44967338fada909de441a084c3e28fa5a26916c85f51d31975` |
| `vial_insertion/model.pth` | `3e5d32b0cc42d37fb3f46258006b9a00352a3901b1860c1b3012c2e62c136522` |

本轮实际使用 multitask 权重。checkpoint 内模型为 `nsm_transformer`，点特征维数 3，4 blocks、1 head、hidden dim 256、输入点数 1024，启用 timestep embedding，最大 timestep 5。实际推理参数为 seed 0、10 refinement iterations、5 initial candidates、无 success classifier。

## 输入、裁剪与坐标检查

真实输入：

- RGB：`runs/grasp_snapshots/20260520_131624_西瓜霜/color.png`，SHA-256 `943be99597a40d75b08a770ccc621cc495a21cc993be7908972a078d69d1e734`
- 深度：`runs/grasp_snapshots/20260520_131624_西瓜霜/depth.png`，SHA-256 `8e84c4e0acbb7a0907493e468ab9af4e6e4be93384fdeecb4a3ccfdd39d2f5c9`
- 历史目标 mask：`runs/grasp_snapshots/20260520_131624_西瓜霜/mask.npy`，SHA-256 `aa94687183e1a69c75cbf41a28965067cd89b99d2052032094cc88e9662c6135`
- 配置：[anyplace_real/config/real_saved_xarm_snapshot.json](anyplace_real/config/real_saved_xarm_snapshot.json)

其中目标使用历史分割 mask，并人工复核深度范围；`base_obj` 是人工指定的空桌面矩形区域 `[500,515]–[1270,715]`。该人工辅助已写入 manifest 和 `preprocess_report.json`。没有使用 Molmo/SAM，也没有伪装成自动分割结果。

坐标处理顺序：

1. 用对齐内参将深度反投影到腕相机 OpenCV optical frame（x 向右、y 向下、z 向前），尺度为米。
2. `wrist_handeye.yaml` 的矩阵按 `T_ee_camera` 解读，即把相机点映射到 EE。
3. 历史 EE pose 按 xArm roll/pitch/yaw 与现有代码一致地构造 `T_xarmbase_ee`。
4. `T_xarmbase_camera = T_xarmbase_ee @ T_ee_camera`，再把两份点云都变换到旧 xArm base 的重力 z-up 坐标系后送入 AnyPlace。

矩阵往返最大误差约 `1.3e-16 m`。真实目标输入点云 2,210 点、尺度约 `[0.1283, 0.0952, 0.0060] m`；人工桌面局部区域 30,000 点、尺度约 `[0.0861, 0.3184, 0.0123] m`。

AnyPlace 的 `base_obj` 始终表示 parent/support（本例为桌面局部区域），**不是机器人 base frame**。两者只是在本例中都用机器人基座坐标表示，语义不可混用。

官方 `inference_official.py` 会在内部中心化点云，最终返回的 `4x4` 矩阵仍按官方可视化脚本的约定直接作用于原始 target 点：`p_placed = R p_original + t`。本轮叠加图也按此方向生成。

公开脚本中的任务裁剪没有被泛化：rack/vialplate/tuberack 分支根据目标最小尺寸和 hole location 使用专用 x/y/z 范围，其中还包含“simulated data”用的 z 偏移。官方 lid/cup smoke test 使用完整 cup 点云；真实测试直接使用人工确认的桌面局部点云，并将 `use_molmo=false`、`parent_crop=false`。这些是本轮实际配置，不是对所有任务的默认建议。

## 实际运行结果与产物

### 官方样例

- 输入：`/home/wpy/anyplace_data/anyplace_eval/task_gpp_stacking/stacking/lid_0_on_cup_0`
- 产物：`runs/anyplace_real_stage1/official_lid0_cup0/`
- 推理主体耗时：约 `4.41 s`
- 输出：5 个变换；候选 0 的旋转行列式约 `0.9999986`，正交误差约 `2.57e-6`。
- 变换后 lid 的 z 范围约 `0.6511–0.6878 m`，cup 的 z 范围约 `0.5359–0.6641 m`；只证明模型 I/O 和变换可视化正常，不代表物理稳定性通过。

### 真实 RGB-D

- 产物：`runs/anyplace_real_stage1/real_saved_xarm_snapshot/`
- 最终重复推理主体耗时：约 `3.53 s`
- 输出：5 个变换；候选 0 的旋转行列式约 `0.9999990`，正交误差约 `2.24e-6`。
- 简化几何检查：目标投影位于人工 support XY AABB 内，但最低点比桌面中位高度低约 `0.0292 m`，未通过最大 `5 mm` 穿透阈值。
- 结论：模型已在真实点云上运行，但候选 **不可执行**。未生成机器人 waypoint；未声明真机任务成功。

关键产物：

- `preprocess_report.json`：输入、人工辅助、尺度和坐标检查；
- `target_obj_pcd.ply`：重力坐标系目标点云；
- `base_obj_pcd.ply`：重力坐标系承载物点云；
- `local_crop_pcd.ply`：实际送入模型的局部区域；
- `placement_transform.npy`：选中候选的原始 AnyPlace 变换；
- `target_obj_pcd_transformed.ply`：变换后的目标点云；
- `placement_overlay.png`：承载物、变换前和变换后的叠加图；
- `inference_report.json`：推理配置、权重 hash、矩阵检查和几何检查；
- `offline_plan.json`：明确禁止执行的离线对象运动提案；
- `repeatability_report.json`：同 seed 重复误差；
- `inference.log`：实际模型运行日志。

### 开发尝试完整记录

1. 未将 AnyPlace 环境的 `bin` 加入 `PATH`：KNN 扩展报告找不到 Ninja；没有模型输出。
2. 加入 `PATH`、未加入环境 CUDA headers：KNN 编译报告 `cuda_runtime_api.h` 缺失；没有模型输出。
3. 加入该独立环境已有的 `PATH/CPATH/LIBRARY_PATH`：KNN JIT 编译成功，没有安装或降级新包。
4. 第一轮真实推理误把 camera optical frame 直接当 AnyPlace world frame。模型虽返回结果，但重力轴语义错误；记录保存在 `development_attempt_camera_frame_report.json`、`development_attempt_camera_frame_overlay.png` 和对应 log，不计为有效真实候选。
5. 修正为旧 xArm base 重力 frame 后推理成功；随后加入承载面检查，再以同参数重跑。
6. 同 seed 再重复一次，变换最大差 `0.0`。最终候选仍因约 29.2 mm 穿透而标记不可执行。

## 运行命令

在项目根目录运行。

准备官方样例：

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.prepare_official \
  --case-dir /home/wpy/anyplace_data/anyplace_eval/task_gpp_stacking/stacking/lid_0_on_cup_0 \
  --output-dir runs/anyplace_real_stage1/official_lid0_cup0
```

准备保存的真实 RGB-D：

```bash
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.prepare_real \
  --manifest anyplace_real/config/real_saved_xarm_snapshot.json \
  --output-dir runs/anyplace_real_stage1/real_saved_xarm_snapshot
```

运行 AnyPlace（把 `--input-dir` 换成官方或真实输出目录）：

```bash
PATH=/home/wpy/.conda/envs/anyplace/bin:$PATH \
CPATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/include \
LIBRARY_PATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/lib \
LD_LIBRARY_PATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/lib \
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.run_inference \
  --input-dir runs/anyplace_real_stage1/real_saved_xarm_snapshot \
  --n-refine-iters 10 --init-k 5 --seed 0
```

查看叠加图：

```bash
xdg-open runs/anyplace_real_stage1/real_saved_xarm_snapshot/placement_overlay.png
```

运行轻量测试：

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m pytest -q tests/test_anyplace_real_geometry.py
```

## 进入首个真机放置的最少条件

1. 先确认实际目标平台：为本项目补齐 Franka + Robotiq 接口，或由负责人明确批准改用当前 xArm + GT200；不能混用两套 TCP 和标定。
2. 恢复并核对腕部 D435i 序列号，采集一组带 aligned intrinsics、明确米单位、RGB/depth 帧号、硬件时间戳和拍摄前后机器人位姿的快照。
3. 实测工具 TCP，记录 `T_flange_tcp`；抓取后建立 `T_tcp_object`。重新验证手眼标定，并记录残差、日期和硬件序列号绑定。
4. 获取足以估计目标完整外形/底面的多视角点云或可信 CAD，不能用当前单面点云进行稳定性和碰撞判断。
5. 对全部候选做承载接触、网格碰撞和稳定性筛选；当前候选必须先消除约 29.2 mm 的估计穿透。
6. 将通过的对象变换转换为 TCP goal，并用实际机器人模型、工具模型和现场碰撞场景完成 IK 与离线轨迹规划；人工复核后才可进入单步、低速、受监控执行阶段。

## 修改文件与回滚

本轮新增：

- `anyplace_real/__init__.py`
- `anyplace_real/geometry.py`
- `anyplace_real/prepare_real.py`
- `anyplace_real/prepare_official.py`
- `anyplace_real/run_inference.py`
- `anyplace_real/config/real_saved_xarm_snapshot.json`
- `anyplace_real/README.md`
- `tests/test_anyplace_real_geometry.py`
- `ANYPLACE_REAL_STAGE1_REPORT.md`

本轮修改：

- `services/realsense_dual_camera_service.py`

精确回滚本轮代码（不会触碰开始前已有的 Gate 改动）：

```bash
git restore -- services/realsense_dual_camera_service.py
git clean -fd -- anyplace_real tests/test_anyplace_real_geometry.py ANYPLACE_REAL_STAGE1_REPORT.md
```

运行产物位于已被 `.gitignore` 忽略的 `runs/anyplace_real_stage1/`；如确认不再需要，可单独删除该明确目录。
