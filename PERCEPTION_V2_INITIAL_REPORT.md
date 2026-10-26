# 药盒 RGB-D 感知 v2 初步优化报告

日期：2026-09-10<br>
项目：`/home/wpy/pythonproject/armcode-agent`<br>
代码基线：`ab7a00b842ba5ac3d2c7fcb0a50eb7f2c48fbfb7`，分支 `phase1-learnable-gate`

## 结论

已建立一条仅离线运行的新感知入口：YOLOE 一次生成全图药盒实例与掩码，MobileCLIP 根据本地参考图确认指定药盒身份，再用实测深度清理掩码并生成相机系/xArm 基座系点云。没有初始化或控制真机，没有修改旧 Gate、旧识别入口或抓取执行参数。

在现有 42 组保存 RGB-D 开发数据上，新 YOLOE 候选对历史目标框的召回为 42/42；若只按检测置信度选择，找对历史目标框为 6/42，加入指定药盒参考图检索后为 42/42。这个结果说明“实例提议 + 身份绑定”在当前已知药盒数据上有效，但不是未知药盒或新环境的独立泛化结果。

## 实施内容

- `perception_v2/yoloe_backend.py`：官方 THU-MIG YOLOE 开放词汇实例检测/分割适配器；
- `perception_v2/retrieval.py`：MobileCLIP 参考图库、掩码裁剪与指定身份评分；
- `perception_v2/geometry.py`：深度单位处理、稳健深度带、最大连通域、抓取掩码、点云和坐标变换；
- `perception_v2/run_snapshot.py`：单帧端到端入口和四联对比图；
- `perception_v2/benchmark_snapshots.py`：保存数据开发集对比入口；
- `tests/test_perception_v2_geometry.py`：几何和旧数据标签兼容单测。

原 `find_and_grasp_medicine.py` 未被替换，因本轮要求先做初步优化和对比，且新身份阈值尚未用独立数据校准。新入口产生的 `grasp_mask.npy` 与点云可在下一阶段以显式开关接入现有 AnyGrasp 服务。

现有主流程审计确认：`skill_lib/combound/find_and_grasp_medicine.py` 会逐条执行多个 GroundingDINO prompt、按框 IoU 融合、逐候选 OCR，并用固定权重打分；SAM 候选分割和把 mask 传给 AnyGrasp 的默认值均为 `False`。因此旧默认路径不能稳定提供“身份确认后的准确实例掩码”。这次新增入口把一次实例分割设为主干，身份检索负责在多个药盒实例中绑定指定 SKU，OCR 后续只需承担低置信复核，而不再是每个候选的必经瓶颈。

## 模型与环境审计

独立环境：`/home/wpy/.conda/envs/armcode_recognition_v2`，约 14 GB。原有机器人环境未降级。

- Python 3.10.20；
- PyTorch 2.5.1+cu121，torchvision 0.20.1+cu121；
- Ultralytics 8.3.39（YOLOE 官方仓库内版本）；
- OpenCV 4.13.0，Open3D 0.19.0，NumPy 2.2.6；
- GPU：NVIDIA RTX 2080 Ti 11 GB；
- YOLOE 仓库 commit：`40cd606cabdbe2b566d6f14a6b162c89206e9a1b`。

YOLOE 仓库当前许可证为 AGPL-3.0。研究复现可继续使用，但若后续把这段实现部署为对外服务或分发，需要先做项目许可证兼容审查。

权重：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `dependencies/yoloe/pretrain/yoloe-v8s-seg.pt` | 31,135,890 B | `ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536` |
| `dependencies/yoloe/mobileclip_blt.pt` | 599,214,572 B | `670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a` |

## 对比结果

### 42 组现有保存数据

完整原始结果：`runs/perception_v2/benchmark_42/benchmark.json`。

| 指标 | 结果 |
|---|---:|
| YOLOE 候选召回，历史目标框 IoU ≥ 0.50 | 42/42 (100%) |
| YOLOE 候选召回，历史目标框 IoU ≥ 0.75 | 42/42 (100%) |
| 仅检测置信度选择，IoU ≥ 0.50 | 6/42 (14.3%) |
| 指定身份参考图检索后选择，IoU ≥ 0.50 | 42/42 (100%) |
| 身份检索 top-1（4 个已有药盒类别） | 42/42 (100%) |
| 最优候选框 IoU 中位数 | 0.841 |
| 身份选择框 IoU 中位数 | 0.840 |
| YOLOE 单帧中位耗时（已热身） | 10.9 ms |
| 身份检索中位耗时（图库已编码） | 35.1 ms |
| 旧掩码有效深度像素比例中位数 | 0.886 |
| 新掩码有效深度像素比例中位数 | 0.962 |
| 旧掩码最大深度连通域比例中位数 | 0.810 |
| 新掩码最大深度连通域比例中位数 | 0.986 |

评价边界：历史框和掩码来自旧流水线，不是人工真值。框 IoU 用于确认是否找回同一任务目标；历史掩码 IoU 只能表示一致性，不能表示分割精度。图库计算排除了当前被测快照，但训练/参考与测试仍来自同一批物理药盒和拍摄环境。

保留的失败尝试：第一次 4 张 smoke benchmark 得到身份选择 0/4。原因不是模型分类失败，而是早期快照的 `grasp_input.json` 没有保存 `query`，评测器实际传入了空身份。已增加旧快照目录名兼容，并在测试中锁定“JSON query 优先、旧目录名仅作离线回放回退”；修复后同 4 张为 4/4。在线接入时必须由任务节点显式提供 query，不能从数据集目录读取标签。

### 与当前 SAM3 bbox 分割路径的速度抽查

在前 8 张保存图上，给 SAM3 提供已有目标框后，逐框分割中位耗时为 1.978 s；YOLOE 在同 8 张图上完成全图实例提议和掩码的中位耗时约 10.2 ms，另加身份检索约 28.2 ms。两者输入条件不同：SAM3 已经拿到目标框，YOLOE 路径还完成了候选产生，因此这里只能作为本机工程吞吐对比，不是模型精度排名。

SAM3 抽查产物：`runs/perception_v2/sam3_baseline_8/timing.json` 和 `quality.json`。RTX 2080 Ti 不支持 Flash Attention，该结果不能外推到更新 GPU。

### 单帧完整产物

输入：`runs/grasp_snapshots/20260520_131624_西瓜霜`<br>
输出：`runs/perception_v2/xiguashuang_v2`

身份检索将目标判为“西瓜霜”，分数 0.920，次高“抗病毒口服液”0.792，差值 0.128。生成 47,852 个抓取掩码点；相机系点云范围约 `93 × 128 × 11 mm`，变换后基座系范围约 `128 × 94 × 12 mm`。RGB 语义掩码完整；深度面和抓取掩码保留了包装反光造成的真实深度空洞，没有伪造深度。

## 坐标、输入和隔离检查

- 彩色图和深度图来自保存快照；深度按快照 `depth_scale` 转成米；
- 使用快照中的对齐深度内参反投影，点坐标为相机光学系 `x右/y下/z前`；
- `T_base_camera = T_base_ee(capture_ee_pose_mrad) @ T_ee_camera`；
- `config/calibration/wrist_handeye.yaml` 的方向会显式解析，单位必须为米；
- 目录/JSON 中的药盒名称只作为请求身份和评价标签，不参与几何、掩码清理或点云构造；
- 历史 bbox/mask 仅进入 benchmark/evaluator，正常运行可不加 `--evaluate-reference-mask`；
- 相机到基座变换已做数据级合理性检查，但本轮没有现场标定板复测，因此不能标记为“真机重新实测通过”。

## 运行命令

单帧和批量命令见 `perception_v2/README.md`。最小测试：

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python -m pytest -q tests/test_perception_v2_geometry.py
```

## 尚缺条件与下一步

进入自动抓取前至少还需要：

1. 用独立拍摄条件和新增 SKU 做人工标注验证，并校准身份接受阈值、top-1/top-2 margin 和拒识策略；
2. 对低置信、遮挡或图库缺失情况保留 OCR/多视角复核，而不是强制选一个实例；
3. 现场复核 D435i/D455 实际序列号、RGB-D 对齐元数据、采集时 EE 位姿同步误差和手眼标定；
4. 将新 `grasp_mask` 接到 AnyGrasp 前跑离线碰撞/工作空间过滤和回放测试；
5. 为透明膜/强反光药盒增加多视角或主动深度补全，但不能把推断深度当成实测点云。

因此，当前已具备“指定已知药盒 → 准确候选掩码 → 深度有效点云”的离线数据生成条件；尚不具备无监督未知药盒拒识和自动真机抓取放行条件。

## 回滚

本次新增内容集中在 `perception_v2/`、`tests/test_perception_v2_geometry.py`、`requirements-perception-v2.txt` 和本报告。删除这些新增文件即可回滚代码；`runs/perception_v2/`、`dependencies/yoloe/` 和独立 Conda 环境均为隔离产物，可按需另行移除。本次没有提交、覆盖或清理用户原有未提交改动。
