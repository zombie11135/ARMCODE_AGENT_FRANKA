# AnyPlace xArm + GT200 Stage 2 报告

日期：2026-09-10（Asia/Shanghai）<br>
项目：`/home/wpy/pythonproject/armcode-agent`<br>
基线提交：`ab7a00b842ba5ac3d2c7fcb0a50eb7f2c48fbfb7`<br>
安全边界：离线计算和只读位姿采样；未下发机械臂运动、夹爪动作、故障复位或控制器参数写入。

## 结论

已经实现三个进入 AnyPlace 真机放置前的最小入口：

1. xArm + GT200 的低成本 TCP/Pivot 位姿采样和离线残差拟合；
2. 多视角 RGB-D 目标点云融合，以及由实测盒体尺寸生成、带明确 provenance 的完整碰撞代理；
3. 对全部 AnyPlace 候选做承载面投影、支撑范围、接触斑块、障碍间距和旋转合法性检查，并在信息齐备时生成不可执行的 TCP waypoint 提案。

另外补齐了每次抓取后的 `T_tcp_object` 多帧估计入口。这是把 AnyPlace 的对象运动变换转换为机器人 TCP goal 的必要条件，不等价于 TCP/Pivot 标定。

当前仍不能执行首个真机放置：已有真实数据只有单视角表面，尚无实测 `T_flange_tcp`、当前抓取的 `T_tcp_object`、完整目标代理、xArm/GT200 碰撞模型和 IK/扫掠碰撞结果。所有规划输出固定 `execution_allowed=false`。

## 已实现入口

### 1. GT200 TCP/Pivot 标定审计

- [anyplace_real/record_tcp_teach_poses.py](anyplace_real/record_tcp_teach_poses.py)：操作者在 xArm Studio 中手动摆姿态并按 Enter 后，只读记录法兰位姿。程序拒绝非零控制器 TCP offset；不调用运动使能、运动、清错、TCP 写入或 `/grip`。
- [anyplace_real/tcp_calibration_audit.py](anyplace_real/tcp_calibration_audit.py)：离线求解 `R_i p_flange_tip + t_i = p_base_pivot`，报告 RMS、最大残差、秩、条件数及姿态覆盖。
- [anyplace_real/config/tcp_pivot_manifest.example.json](anyplace_real/config/tcp_pivot_manifest.example.json)：默认 `example_only=true`，防止把示例值误当实测值。

Pivot 只能确定“法兰坐标系中固定尖端的位置”。若临时指针尖端不是希望使用的 GT200 抓取中心 TCP，必须沿已定义的工具轴减去尺量的尖端伸出量。Pivot 本身不能确定 TCP 轴方向；方向需要依据 GT200 机械对称轴定义，并通过已知平面上的平移/姿态检查独立验证。因此程序在 `tcp_orientation_rpy_rad=null` 时只输出 position-only 结果，不生成权威 `T_flange_tcp.npy`。

### 2. 多视角目标几何

- [anyplace_real/multiview_fusion.py](anyplace_real/multiview_fusion.py)：读取已保存且对齐的 RGB-D、目标 mask、每帧内参和拍摄时变换，统一变换到 robot base 后体素融合。
- [anyplace_real/config/multiview_manifest.example.json](anyplace_real/config/multiview_manifest.example.json)：显式区分直接 `T_base_camera` 与 `T_base_flange @ T_flange_camera` 两种输入。
- [anyplace_real/geometry.py](anyplace_real/geometry.py)：增加体素降采样、桌面盒体拟合和完整盒体表面采样。

输出同时保留 `observed_fused_cloud` 与 `complete_collision_proxy`。后者只有在填入人工实测长宽高后才生成，并标为 measured-dimensions proxy；不会把不可见表面伪装成深度观测。当前 `runs/grasp_snapshots/` 中的历史图像拍摄位姿几乎相同，不构成有效多视角数据。

### 3. AnyPlace 全候选几何过滤与 TCP goal 组合

- [anyplace_real/candidate_filter.py](anyplace_real/candidate_filter.py)：不再只取候选 0；依次检查所有候选的旋转矩阵、重力方向落面修正、支撑凸包内比例、几何中心支撑、底部接触斑块及可选障碍间距。
- 若提供 `T_base_object_input` 和本次抓取的 `T_tcp_object`，按以下关系组合：

  `T_base_object_goal = T_object_motion @ T_base_object_input`

  `T_base_tcp_goal = T_base_object_goal @ inverse(T_tcp_object)`

- 输出 pre-place/place/retreat 矩阵仅是离线 waypoint proposal；由于还没有真实工具模型 IK 和完整扫掠碰撞检查，代码固定禁止执行。

### 必要补充：每次抓取后的 `T_tcp_object`

- [anyplace_real/held_object_relation.py](anyplace_real/held_object_relation.py)：用至少三组同步 `T_base_tcp` 与同一对象 frame 的 `T_base_object`，计算并平均 `inverse(T_base_tcp) @ T_base_object`，报告平移和旋转散布。
- [anyplace_real/config/held_object_manifest.example.json](anyplace_real/config/held_object_manifest.example.json)：显式记录矩阵方向和单位，默认拒绝示例。

该关系是 per-grasp 状态；物体滑动或重新抓取后必须失效重测。腕部相机与夹爪刚性同动，移动机械臂不会产生新的 camera-to-held-object 相对视角；需要观察更多被夹物表面时应使用固定外部相机、抓前完整几何或实测尺寸代理。

## 无专用设备时怎样做 Pivot

最低成本材料是一根直而硬的笔杆、竹签或细杆，纸、胶带和尺子。不需要四点标定球：

1. 把杆刚性夹持或固定在 GT200 中线上，测量“希望使用的抓取中心”到杆尖的轴向伸出量；不要用软线或会弯的吸管。
2. 把纸固定到刚性桌面，画一个小十字；更稳妥的是在橡皮、软木或厚纸上做一个很浅的小凹点，使尖端能重复回到同一点而不施加明显力。
3. 在 xArm Studio 的手动/示教模式中低速移动，尖端轻触同一点。改变腕部方向而尽量保持尖端位置不变，共记录 8--12 个姿态；整体姿态跨度至少 30 度，并保留 1--2 个姿态只做验证。
4. 拟合后先看 RMS/max residual。纸笔方案把 `2 mm RMS / 4 mm max` 作为初期审计阈值，不作为亚毫米精密标定声明。
5. 用尺量伸出量把“指针尖端 TCP”换算到目标抓取中心，并从 GT200 的开合轴、前向轴定义 RPY。再用桌面已知方向做独立检查。

若连硬质笔杆都没有，只能用卡尺/直尺从法兰安装面量到闭合夹爪中心，并生成 `approximate_mechanical_measurement` 配置；它适合检查厘米级错误，不应直接用于贴边、插入或小余量放置。最值得添置的低成本物品不是昂贵标定球，而是刚性细杆、简单的 3D 打印指针和一把卡尺。

安全要求：只在手动/示教模式低速操作，保持急停可及，不用尖锐金属针，不让机械臂自动追踪接触点，不对桌面持续施力。

## 实际运行与验证

测试命令：

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/anyplace/bin/python -m pytest -q \
  tests/test_anyplace_real_geometry.py tests/test_anyplace_real_stage2.py
```

实际结果：`8 passed in 0.20s`。覆盖合成 Pivot 真值恢复、完整 TCP 结果写出、体素融合、实测尺寸盒体代理、多视角端到端数据流、`T_tcp_object` 平均，以及候选落面/支撑过滤。`anyplace` Python 3.8 与项目 Python 3.10 的 `compileall` 均通过，`git diff --check` 通过。

使用 Stage 1 真实 AnyPlace 的 5 个候选和当前单视角目标点云做了保守 smoke test：5 个候选均因“接触点斑块不能支撑几何中心”未通过，重力落面修正约为 `24.5--30.8 mm`；结果在 `runs/anyplace_real_stage2/current_partial_cloud_smoke/candidate_filter_report.json`。该测试输入不是完整 collision proxy，因此结论是“当前输入不足以确认候选稳定”，不是“AnyPlace 模型失败”。

开发过程还记录了一次无效尝试：首次把 stdout 重定向到尚未创建的 `runs/anyplace_real_stage2/`，shell 在 Python 启动前失败；创建明确输出目录后重跑成功。没有遗漏为成功案例。

## 用户运行命令

### A. 只读记录手动 Pivot 姿态

先确保控制器 TCP offset 为零，机械臂由操作者在 xArm Studio 手动摆放：

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.record_tcp_teach_poses \
  --xarm-ip 192.168.1.237 --count 10 \
  --output runs/anyplace_real_stage2/tcp/teaching_poses.json
```

### B. 离线拟合 TCP 位置

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.tcp_calibration_audit \
  --manifest runs/anyplace_real_stage2/tcp/teaching_poses.json \
  --output-dir runs/anyplace_real_stage2/tcp/audit
```

在独立核对 GT200 工具轴以前，不要填写 `tcp_orientation_rpy_rad`，也不要把 position-only 结果写入控制器。本项目没有提供任何写控制器 TCP 的命令。

### C. 融合已保存多视角 RGB-D

复制示例配置，替换真实文件、姿态、相机参数和药盒实测尺寸，并把 `example_only` 改为 `false`：

```bash
cp anyplace_real/config/multiview_manifest.example.json \
  anyplace_real/config/my_multiview_manifest.json
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.multiview_fusion \
  --manifest anyplace_real/config/my_multiview_manifest.json \
  --output-dir runs/anyplace_real_stage2/multiview
```

### D. 求本次抓取的 `T_tcp_object`

```bash
cp anyplace_real/config/held_object_manifest.example.json \
  anyplace_real/config/my_held_object_manifest.json
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.held_object_relation \
  --manifest anyplace_real/config/my_held_object_manifest.json \
  --output-dir runs/anyplace_real_stage2/held
```

### E. 离线过滤所有 AnyPlace 候选

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.candidate_filter \
  --predictions runs/anyplace_real_stage1/real_saved_xarm_snapshot/anyplace_stage1_relative_pose_prediction_real.npy \
  --target-proxy runs/anyplace_real_stage2/multiview/complete_collision_proxy.npy \
  --support runs/anyplace_real_stage1/real_saved_xarm_snapshot/base_obj_pcd.npy \
  --T-base-object-input runs/anyplace_real_stage2/multiview/T_base_object_input.npy \
  --T-tcp-object runs/anyplace_real_stage2/held/T_tcp_object.npy \
  --output-dir runs/anyplace_real_stage2/candidate_filter
```

## 尚缺的最少条件

1. 按上述流程采 8--12 个 GT200 指针姿态，得到有残差记录的 TCP 位置，并独立定义/验证 TCP 方向；必要时重新验证工具变更后的手眼链。
2. 获取真正不同视角的目标数据，或量取当前药盒长宽高生成完整代理；不能继续把单面点云当完整物体。
3. 每次抓取后同步估计 `T_tcp_object`，并验证其多帧散布在容差内。
4. 把 xArm 6、GT200、当前 TCP、桌面和障碍物模型接入 IK 与全轨迹扫掠碰撞检查；加入关节限位、奇异性和线缆/腕部相机包络。
5. 在执行程序之外先人工检查候选、预放置和退离路径。只有以上全部通过，才能另开受监控、低速、单步的真机执行阶段。

## 修改文件与回滚

Stage 2 新增/修改集中在：

- `anyplace_real/geometry.py`
- `anyplace_real/tcp_calibration_audit.py`
- `anyplace_real/record_tcp_teach_poses.py`
- `anyplace_real/multiview_fusion.py`
- `anyplace_real/candidate_filter.py`
- `anyplace_real/held_object_relation.py`
- `anyplace_real/config/tcp_pivot_manifest.example.json`
- `anyplace_real/config/multiview_manifest.example.json`
- `anyplace_real/config/held_object_manifest.example.json`
- `anyplace_real/README.md`
- `tests/test_anyplace_real_stage2.py`
- `ANYPLACE_REAL_STAGE2_REPORT.md`

这些文件目前未提交；删除 Stage 2 独立新增文件并恢复 Stage 1 版本的 `geometry.py`、`README.md` 即可回滚。由于工作树原本已有用户修改，不建议使用宽泛的 `git clean`、`reset` 或目录级 `restore`。运行产物在明确目录 `runs/anyplace_real_stage2/`，确认不再需要后可单独移走或删除。
