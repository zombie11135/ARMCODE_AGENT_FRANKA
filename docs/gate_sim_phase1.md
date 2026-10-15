# GateOccludedPickPlace-v0：阶段一交付报告

## 结论

阶段一实验底座已经实现并通过最小验收。它是单环境、CPU、PyBullet DIRECT
仿真，使用 Panda、桌面、程序化盒体、浅容器和单容量 buffer。没有连接真机，
也没有引入训练、LLM、付费 API 或大型视觉模型。

## 带贴图日常物品扩展（scene config v3）

默认场景现加入四个外围日常杂物：带包装贴图的燕麦盒、橙色饮料罐、紫色杯状
容器和青色水瓶。盒体使用 box 碰撞，后三者使用 cylinder 碰撞；全部具有质量、
摩擦、接触和沉降过程。贴图由 Pillow 在本地确定性生成，版本为
`procedural-daily-products-v1`，不使用网络或大型资产包。

这些对象不是只用于背景渲染：当前相机 RGB-D 的颜色区域和深度反投影会生成对应
`Estimate`，策略不读取 PyBullet segmentation/body id。物理真值、材质来源和运动状态
只保留在 evaluator。它们的默认位置在任务搜索区外围，以增加外观和形状变化而不改变
A/B/C/D 原来的遮挡定义；后续训练可通过配置修改对象目录和摆放分布。若要复现实验
底座的纯色简版，可构造 `GateSimConfig(textured_daily_clutter_enabled=False)`。

生成四类初始视角图库：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli render \
  --output-dir runs/gate_sim/textured_gallery \
  --width 1280 --height 720
```

直接打开拼图：

```bash
xdg-open runs/gate_sim/textured_gallery/textured_scene_gallery.png
xdg-open runs/gate_sim/textured_gallery/overview_scene_D_1280x720.png
```

录制带贴图场景 C 的物理清障视频：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli manip-video \
  --output-dir runs/gate_sim/manipulation_video \
  --width 1280 --height 720
```

场景 E 将日常物品设为真正任务目标：带贴图燕麦盒被蓝色方块覆盖，先将蓝色方块
搬到 buffer，再把燕麦盒抓放至紫色目标容器。为四个 Gate 输出和完整链路分别录像：

```bash
.venv-gate-sim/bin/python -m gate_sim.cli gate-videos \
  --output-dir runs/gate_sim/gate_action_videos \
  --width 1280 --height 720
```

四个动作的含义保持原定义：`EXECUTE` 视频包含目标抓取与放置；
`VISUAL_PROBE` 只移动观察位姿并获得新 RGB-D；`MANIPULATION_PROBE` 只清障并
重新观察；`REPLAN` 因尚未接入仿真规划器而是带文字说明的静态无动作视频。

实际验收图和报告位于 `runs/gate_sim/textured_clutter_dev/` 和
`runs/gate_sim/gate_action_videos/`；扩展后 8 项仿真定向测试、45 项 Gate 相关测试、
三个原物理 demo、场景 E 完整任务和快照/分支恢复均通过。

三个固定 demo 均成功：

- A/direct：目标从当前 RGB-D 确认后，物理抓取、搬运、释放到容器；
- B/observe_first：初始视角不满足执行条件，机械臂移动到第二个合法观察位姿，
  获取一张新的 RGB-D 后完成抓放；
- C/observe_then_manipulate：换视角后仍不足，固定候选选择器从策略记忆选中蓝色盒，
  双指接触后搬到 buffer，重新拍摄，再完成目标抓放。

稳定视频引用：

- `runs/gate_sim/acceptance/videos/demo_A_direct.mp4`
- `runs/gate_sim/acceptance/videos/demo_B_visual_probe.mp4`
- `runs/gate_sim/acceptance/videos/demo_C_manipulation_probe.mp4`

机器可读报告：

- `runs/gate_sim/acceptance/demo_report.json`
- `runs/gate_sim/acceptance/restore_report.json`
- `runs/gate_sim/acceptance/comparison_report.json`

`runs/` 已被仓库忽略；这些是本机验收产物，不会被误提交为源码。

## 审计与保护

- 审计时仓库位于 `phase1-learnable-gate` 分支，工作树干净；新增实现放在独立
  `gate_sim/` 子目录。
- 现有 Gate 实际包含 legacy、learned、shadow、`GateInput/GateOutput`、
  `decision_id` 关联日志。learned schema 是固定 23 维。
- 未改动旧 Gate 公式、0.7 默认阈值、默认模式、23 维顺序或 MLP 结构。
- 现有执行底座是 xArm/D435i/AnyGrasp 真机链路，不能安全复用于仿真；本包没有
  导入 `services/` 或 `skill_lib/`，`real_robot_enabled=false`。
- 审计时 NVIDIA 驱动不可用；本环境不依赖 GPU。
- 后端选择有一项明确偏离建议：项目内没有 ManiSkill/SAPIEN，当前机器的 NVIDIA 驱动
  也不可用。为优先完成可重复物理执行与恢复验收，采用了主机上已有可运行先例的
  PyBullet CPU 后端，并在本项目环境中独立锁版；没有假装这是 ManiSkill 3。
- 依赖只装入项目本地 `.venv-gate-sim`。核心锁定版本：Python 3.9.18、
  PyBullet 3.2.7、NumPy 1.26.4、Pillow 10.4.0、imageio 2.35.1、
  imageio-ffmpeg 0.5.1、pytest 8.3.5。

## 接口和动作语义

`GateOccludedPickPlaceEnv` 提供：

```python
reset(seed, scenario_type=None)
get_gate_observation()
step_gate(action)
save_snapshot()
restore_snapshot(snapshot, branch_id=None)
```

`step_gate` 每次执行完整宏动作，而非单个关节周期：

- `EXECUTE`：只在当前 RGB-D 记忆存在有效目标估计时抓取；持有目标时放入容器并
  释放。信息不足时不可用，`primitive_success=null`。
- `VISUAL_PROBE`：机械臂经关节空间安全过渡移动到未访问的固定观察位姿，然后
  渲染新的 RGB-D。未选择视角从不提前渲染或融合。
- `MANIPULATION_PROBE`：固定选择器仅对当前、非 stale、可见、可操作的非目标估计
  按搜索区距离排序；随后抓取、放 buffer、返回当前观察位姿并获取新 RGB-D。
- `REPLAN`：枚举保留，但当前没有可复用的仿真重规划器，因此 `action_mask=false`；
  它不会成功、重置或免费恢复。

任务/仿真时间上限、稳定时长、速度阈值、容器边界、动作预算和相机位姿均在
`gate_sim/config.py` 固定。随机种子、场景版本和完整配置写入每个 episode manifest。

## 物理执行

所有可动物体由 PyBullet 刚体和碰撞形状产生接触与运动。初始化先做非穿透检查，
再沉降 360 个 240 Hz step 并检查线/角速度。抓取轨迹使用 Panda IK、位置控制、
张开/闭合双指和抬升反馈。

为了让 DIRECT 后端的刚性夹持可重复，执行器仅在左右手指同时接触同一个动态物体
后建立固定“接触锁定”约束；放置时先移除约束，再张开手指并让物体自由沉降。
没有接触就失败，不会瞬移物体、写成功标志或关闭碰撞。日志显式记录
`bilateral_contact`、`contact_latch`、抬升高度、最终落点与速度。正常双指抓取接触
记录为 `normal_grasp_contacts`，非目标物体超过 3.5 cm 的非预期位移另记为
`unexpected_disturbances`，不会把一切接触当失败。

没有普通模式的真值位姿抓取分支，也没有启用 `oracle_debug`。

## 观测、记忆与评价真值隔离

策略通道只包含当前实际 RGB-D、机器人自身状态、固定区域、颜色分割/深度反投影得到
的对象估计、历史 Memory 和有效性标记。感知代码不读取 PyBullet segmentation id。
相机或物体变化后，旧 pose/occlusion 字段先标 stale；只有新图像中再次检测到的对象
才更新。搬走物体本身不会直接令目标“可见”。

评价通道保存 PyBullet body 位姿、速度、目标是否在容器、是否释放和稳定时间，只用于
outcome 指标。固定候选选择器源码和测试均检查不引用 evaluator、body id、场景类别或
`true_blocker`。执行器以 RGB-D 估计坐标规划；仿真接触 body id 只用于接触反馈和
outcome 中“抓到请求对象”的评价，不反向进入候选选择或 Gate 输入。

现有 23 维顺序直接从 `nesyro_gate.learnable_gate.GATE_FEATURE_NAMES` 读取并断言一致。
每条记录同时保存 raw value、validity、来源和单位。OCR、VLM、真实信息增益等当前不能
可靠计算的值保持 `null/false`，不伪造分数。

## 日志与恢复

每个宏动作写一条 `gate_decision` 和一条同 `decision_id` 的 `gate_outcome`。日志含
episode/seed/场景版本、动作前后观测与 Memory 引用、候选、mask、动作、执行器反馈、
耗时、计数、评价指标、RGB-D 和 pre-action `.bullet`/metadata 快照引用。未执行动作的
primitive 结果保持 `null`。最终产物审计了 97 对 decision/outcome，缺失关联为 0。

内存快照覆盖物理状态、机器人关节控制 target、相机、任务节点、Memory、计数器、
RNG、持物约束、仿真时间/剩余预算和视频帧游标。恢复测试结果：

| 动作 | 最大状态绝对误差 | RGB 最大像素误差 | 结果一致 | 通过 |
|---|---:|---:|---|---|
| VISUAL_PROBE | 8.908954e-5 | 0 | 是 | 是 |
| MANIPULATION_PROBE | 0 | 0 | 是 | 是 |

状态容差固定为 `1e-4`，RGB 容差为 0，因此 `fork_evaluation_ready=true`。通过后从同一
seed-702/C 快照分别走 visual-first 和 manipulation-first；两分支恢复时 Memory 完全
一致、计数器独立、split group 相同且没有跨数据划分，两者使用相同的 4 宏动作上限和
45 s 仿真预算。

## 小规模固定种子比较

对 seeds 0..7、四种方法共 32 次尝试全部写入报告，没有筛选失败案例。所有方法共享
相同环境、感知、Panda 执行器、约束和预算：

| 方法 | 成功/尝试 | 成功率 | 平均观察数 | 平均操作 Probe 数 |
|---|---:|---:|---:|---:|
| direct | 4/8 | 0.50 | 1.00 | 0.00 |
| observe_first | 6/8 | 0.75 | 2.00 | 0.00 |
| observe_then_manipulate | 8/8 | 1.00 | 2.25 | 0.25 |
| legacy | 0/8 | 0.00 | 2.00 | 0.00 |

legacy 确实通过 `UnifiedNeSyRoGate(mode="legacy")` 接入原门控和原阈值。其结果不能解读
为 Gate 优劣结论：简单颜色感知没有 OCR/VLM，两字段按 legacy 适配器的保守缺失值 0
处理，旧 CSC 门因此持续请求观察或 REPLAN。这一限制没有通过伪造高分绕过。本轮没有
搜索奖励权重，也没有从 A/B/C/D 直接生成动作标签。

## 回答验收问题

1. **是否有真正的物理抓放与视角变化？** 有。Panda 逐步关节/IK 控制、双指碰撞反馈、
   动态刚体搬运与释放；视觉 Probe 物理移动到不同位姿后重新渲染 RGB-D。
2. **Gate、候选选择器和执行器是否使用隐藏真值？** Gate 和候选选择器没有；执行轨迹
   使用 RGB-D 位姿。执行器读取接触 id 只做物理夹持反馈，真实 object mapping、容器
   成功和扰动仅在 evaluator/outcome。
3. **各动作是否产生不同观测或物理效果？** 有。visual 的 observation index/camera/RGB-D
   变化；manipulation 改变动态物体位置、占用 buffer 并重新观察；execute 改变 task node
   并最终让目标在容器稳定释放；replan 不可用且无副作用。
4. **快照恢复是否通过？** 通过，见上表；`fork_evaluation_ready=true`。
5. **缺失字段及后续数据条件？** OCR/VLM、可靠真实信息增益、通用遮挡比例、学习式
   grasp confidence、仿真重规划仍缺失。现有原始 RGB-D、23 维 raw/validity/provenance、
   decision/outcome、物理反馈和可恢复分支已足以采集第一版门控数据，但 legacy 的公平语义
   对比需先补一个共享的非伪造语义感知基线。

## 运行命令

从仓库根目录执行：

```bash
python3 -m venv .venv-gate-sim
.venv-gate-sim/bin/python -m pip install -r requirements-gate-sim.txt

.venv-gate-sim/bin/python -m gate_sim.cli audit
.venv-gate-sim/bin/python -m gate_sim.cli render --output-dir runs/gate_sim/textured_gallery
.venv-gate-sim/bin/python -m gate_sim.cli demo --output-dir runs/gate_sim/acceptance
.venv-gate-sim/bin/python -m gate_sim.cli restore --output-dir runs/gate_sim/acceptance
.venv-gate-sim/bin/python -m gate_sim.cli compare --seeds 8 --output-dir runs/gate_sim/acceptance
.venv-gate-sim/bin/python -m gate_sim.cli validate --seeds 8 --output-dir runs/gate_sim/acceptance

.venv-gate-sim/bin/python -m pytest -q \
  tests/test_gate_sim.py tests/test_gate.py tests/test_gate_occlusion.py \
  tests/test_gate_replanning_constraints.py tests/test_scene_memory_builder.py \
  tests/test_legacy_gate_equivalence.py tests/test_learnable_gate.py \
  tests/test_gate_experience_logging.py tests/test_gate_runtime_integration.py
```

定向测试结果为 43 passed。裸跑仓库全量 `pytest` 会收集 vendored MinkowskiEngine、
真机 `skill_lab` 和 `test_xarm.py`，当前独立环境没有 torch/cv2/xarm/requests，因此在
collection 阶段有 27 个依赖错误；这不是本次代码回归，且未为跑真机测试污染仿真环境。

## 修改与回滚

新增：`gate_sim/`、`tests/test_gate_sim.py`、`requirements-gate-sim.txt` 和本报告；修改
`.gitignore` 与 README 的仿真入口。没有修改 `nesyro_gate/`、`skill_lib/`、真机配置或
其他项目。

尚未提交时可先备份需要的 `runs/gate_sim/`，再只对以下明确目标回滚：

```bash
git restore -- .gitignore README.md
rm -r gate_sim tests/test_gate_sim.py requirements-gate-sim.txt docs/gate_sim_phase1.md .venv-gate-sim
```

若之后单独提交了这些改动，优先使用 `git revert <该提交>`，不要 reset/clean 整个仓库。
