# MobileCLIP2 接入状态（2026-10-22）

用户已确认使用未经额外微调的官方模型。本机已用官方权重和现有参考图重建特征库，并将 agent 的默认识别配置切换到 MobileCLIP2-B。已完成保存照片上的真实模型推理及 RGB-D 遮挡分析回放，未连接相机或执行机械臂动作。

## 接入位置

当前分支已经合并同学的识别实现，流程如下：

```text
腕部 RGB-D → 冻结同一帧
  ├─ RGB → GroundingDINO 候选框 → MobileCLIP2-B 图库匹配 → 目标身份与框
  └─ RGB-D → SAM3 可见区域 → 单视角遮挡关系图
目标身份与框 → 关联关系图节点 → 判断是否清障
需要清障 → 搬运遮挡物 → 旧观测失效 → 重新拍照与识别
目标满足抓取条件 → 原帧深度与目标 mask → 抓取 → 放入容器
```

`services/reference_encoder.py` 定义第二代编码器，`services/object_reference_service.py` 管理参考图和特征；`FindAndGraspMedicineSkill._recognize_image` 是识别入口。`skill_lib/actions/geometry_occlusion.py` 已通过该入口获取身份，再绑定到同一帧的关系图，调用方无需另写识别流程。

MobileCLIP2 负责候选区域的身份匹配。候选框检测、可见区域分割、深度和抓取规划仍分别使用原有服务。目标不在图库或匹配不确定时，`reference_only` 策略不会自动调用 OCR；关系图可以继续产生，但不能将“有关系图”当成“已识别目标”。

## 本机配置与重建结果

- 已创建 `.venv-mobileclip2-fr3`，继承本机原机械臂 Python 环境，仅在隔离环境安装 `open_clip_torch==3.3.0`、`timm==1.0.30`。原 Torch 和硬件环境未替换。
- 已验证新环境中的真实 DINO、MobileCLIP2-B、SAM3 推理，以及相机库、串口库导入。没有以测试替身代替这些离线模型验证。
- 官方 `mobileclip2_b.pt` 位于 `models/mobileclip2-b/`，SHA-256 为 `144b1344f5f01f08af8be1077f0db5eec95bf0fd1e349c1e9ca6102724448486`，与同学文档的记录一致。
- 重建文件位于 `data/object_gallery/vector_libraries/mobileclip2_b/index.npz` 和同目录的 `metadata.json`。4 类、45 张参考图对应 `45 × 512` 的单位向量矩阵：布洛芬 20 张、蒙脱石散 7 张、西瓜霜 12 张、抗病毒口服液 6 张。
- 导出前检查权重与每张参考图的哈希，导出后用正式加载器校验模型、预处理、文件哈希、向量归一化和逐行身份，再发布完整目录。第一代特征缓存没有删除或覆盖。
- `config/perception/references.yaml` 已选择 `encoder_model: mobileclip2_b`、`recognition_policy: reference_only`、相似度门槛 0.75、类别差距门槛 0.02、每类最相似 3 张参考图。相似度按用户要求由最初的 0.80 下调；共享配置模板仍保留原始 0.80，另一台电脑如需复现本次设置，应修改本机 `references.yaml`。
- `config/local.env` 中 `FR3_AGENT_PYTHON` 与 `ARM_AGENT_PYTHON` 的默认值已指向 `.venv-mobileclip2-fr3/bin/python`；抓取独立工作进程的解释器配置保留。已验证启动脚本加载环境后能读到第二代模型及 `45 × 512` 特征。

官方模型来源：[apple/MobileCLIP2-B](https://huggingface.co/apple/MobileCLIP2-B)，下载版本为 `c61288acd0f82f59f99bc4b8691ba1c2c100597a`。

本机配置、模型、特征库、隔离环境不随 Git 提交迁移，另一台电脑需要另行准备。`.venv-mobileclip2-fr3` 继承本机机械臂 Python 环境，也不能直接复制到另一台电脑。

配置对新启动的 agent 进程生效。已运行的进程需要重启；如果当前终端已导出旧解释器路径，先执行 `unset FR3_AGENT_PYTHON ARM_AGENT_PYTHON`，再按原来的命令启动。配置原件保存在 `runs/mobileclip2_rebuild/20261022_191750/before/`，需要回退时恢复其中的 `references.yaml` 和 `local.env` 到原路径。

## 重建与离线复现

在模型、图库和依赖已经准备好的电脑上，从项目根目录执行：

```bash
.venv-mobileclip2-fr3/bin/python scripts/rebuild_reference_bank.py \
  --config config/perception/references.example.yaml \
  --expected-checkpoint-sha256 144b1344f5f01f08af8be1077f0db5eec95bf0fd1e349c1e9ca6102724448486
```

脚本不覆盖已经存在的输出目录。当前电脑已经完成重建，不必重复运行；如需另存新版，传入 `--output` 指定新目录，并同步配置中的 `vector_library_dir`。配置中的相对路径以配置文件所在目录为准。新增参考图后，正式服务也会按需重建模型专属缓存。

本次照片回放命令：

```bash
.venv-mobileclip2-fr3/bin/python -B skill_lab/test_observe_gallery_offline.py \
  --images runs/mobileclip2_rebuild/20261022_191750/images \
  --dino-model /home/seur/data/wpy/models/grounding-dino-tiny \
  --output runs/mobileclip2_rebuild/20261022_191750/replay
```

脚本不启动相机或机械臂，检查每张新照片只做一次 DINO 检测与一次候选批量编码，并记录接受、拒绝和耗时。没有完整人工标签时，这些数量不能作为识别准确率。

## 本机真实模型回放结果

以下为初次重建、门槛为 0.80 时的记录：

- 两张已保存的腕部照片，每张查询 4 类，共 8 次：接受 1 次、拒绝 7 次。每张照片各 1 次 DINO 检测、1 批候选编码，全程不调用 OCR。
- 早期照片中的西瓜霜匹配分数为 0.8493、类别差距为 0.1217，身份通过；在对应 RGB-D 关系图中关联目标节点，框 IoU 为 0.9418，未发现该目标的遮挡证据。
- 遮挡照片中的抗病毒口服液最高查询分数为 0.7925，低于 0.80，身份被拒绝；SAM3 仍产生 8 个可见节点，关系图记录 `target_not_identified`，未把其他节点当成目标。此回放未开启未知目标的探索性清障。
- 这些结果表明重建特征库、身份与几何接口可用，不代表遮挡目标已经可靠识别。阈值未因本次照片而下调，仍需独立场景评估及针对遮挡、视角的参考图补充。

结果目录：`runs/mobileclip2_rebuild/20261022_191750/`。`replay/results.json` 包含每次查询证据，`geometry_summary.json` 包含真实 RGB-D 与 SAM3 关联结果，`activation.json` 记录当前配置、模型哈希及备份位置。回放保留了原照片和原采集时间，不作为当前现场的新鲜观测，也没有测试实时位姿或运动。

2026-10-22 按用户要求将本机相似度门槛下调至 0.75，类别差距仍为 0.02，不重新编码参考图。同样两张照片的 8 次查询中，接受 3 次、拒绝 5 次；新增接受了早期照片中 0.7641 和遮挡照片中 0.7925 的抗病毒口服液候选。这仍不是识别准确率。

检查候选框发现，早期照片中新增的抗病毒口服液框 `[305, 206, 400, 259]` 实际落在西瓜霜盒的局部区域，属于误匹配；遮挡照片中的候选虽然通过身份门槛，与可见节点的关联也仍为 `target_association_ambiguous`。这两个新增候选均未通过几何关联检查，不能直接用于抓取。真机实验应先检查当前照片的识别框及关系图，不能将接受次数增加当作准确率提升。

此次配置备份及回放位于 `runs/mobileclip2_threshold/20261022_192538/`：`references.before.yaml` 为原配置，`replay/results.json` 为新门槛下的真实推理，`geometry_association.json` 为身份框在原帧可见节点上的关联与清障候选判断。后者复用已保存且哈希匹配的掩码，不是当前现场的新鲜抓取观测。尚未开始本轮真机采集或运动实验；运行中的 agent 需要重启才会读取新门槛。

## 本次补充的集成验证

`tests/test_geometry_occlusion_agent.py::test_mobileclip2_identity_drives_geometry_agent_without_ocr` 使用真实识别调度、图库服务、关系图和 agent 流程，模型编码器、相机与执行器使用测试替身：

- 已登记目标的身份进入关系图，执行模拟清障后重新识别，再完成模拟抓取和放置；OCR 调用为零。
- 未登记目标仍保留几何关系图；在未启用未知目标探索时，拒绝清障与抓取，不用旧身份替代。

这验证软件接口和控制流程，不代表 MobileCLIP2 的实测准确率，也不代表真机运动测试通过。

另补充了重建库的逐行身份重排验证，以及权重/参考图变化时编码前拒绝的测试。本次在 `.venv-mobileclip2-fr3` 执行 `python scripts/check_core.py`：426 项通过，2 项跳过。

## 原 agent 交互入口与本次清障缓冲位

用户最新指定遮挡物释放时的基座系**法兰** XYZ 为 `(450, -400, 300)` mm，替换之前的 `(450, -300, 200)` mm；目标药品最终仍放入视觉识别的蓝色盘子。此参数直接指定法兰释放位置，不是桌面点或 TCP 坐标。

`--geometry_occlusion_buffer_xyz_mm 450 -400 300` 已接入原 agent，运行时用本轮实际法兰朝向补齐六维缓冲位姿。即使配置了另一个固定 home，缓冲朝向也读取本轮实际姿态。该三维参数只支持 FR3；已有六维参数保持可用，不会改写输入配置。清障释放点上方 120 mm 为默认接近位，因此本次接近法兰 XYZ 为 `(450, -400, 420)` mm。

从项目目录启动，不提供 `--goal` 即进入原交互界面：

```bash
export FR3_AGENT_PYTHON="$PWD/.venv-mobileclip2-fr3/bin/python"
bash run_fr3_agent.sh \
  --use_llm_planner \
  --enable_geometry_occlusion \
  --disable_geometry_occlusion_auxiliary \
  --enable_occlusion_recovery \
  --occlusion_allow_unlocalized_target \
  --occlusion_recovery_max_attempts 1 \
  --geometry_occlusion_buffer_xyz_mm 450 -400 300 \
  --arm-speed 0.10 \
  --execute_robot
```

在 `armcode>` 输入 `将西瓜霜放到蓝色盘子中`。由同一 agent 完成规划、技能调度、清障和重观测，不需要单独执行采集或清障脚本。此命令允许实际机械臂与夹爪动作，本次开发验证未运行它。

新增 XYZ 参数的 CLI 传递、毫米单位保留、实际朝向绑定、固定 home 与当前朝向不同时的处理均已通过离线测试；本轮完整回归为 429 项通过、2 项跳过。

后续已加入可选的单步 LLM 遮挡恢复，复用本机 planner 模型并在同一个 agent 任务内执行。
启用参数、真实模型回放结果及当前边界见 [LLM 恢复说明](llm_occlusion_recovery.md)。默认不启用该新功能。
