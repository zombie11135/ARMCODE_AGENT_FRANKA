# 药盒 RGB-D 感知 v2（离线）

> 源码发布说明：以下为已有模块和实验环境的使用记录。绝对路径需要按本机修改，模型、外部源码、照片、标定与实测 manifest 不随仓库分发。新成员先阅读 [SETUP](../docs/SETUP.md) 与 [第三方依赖](../THIRD_PARTY.md)。

这个入口把“找到指定药盒”扩展为可直接供抓取模块消费的完整产物：实例语义掩码、深度有效可见面掩码、保守抓取掩码，以及相机系和 xArm 基座系点云。

当前流水线：

1. 官方 YOLOE 对整帧只做一次开放词汇实例检测与分割；
2. MobileCLIP 将每个实例与本地同药盒历史参考图匹配，把用户指定名称绑定到实例；
3. 用 RGB-D 深度中位数/MAD、一致深度带和最大连通域清除桌面泄漏；
4. 对可见面掩码做保守腐蚀，使用对齐深度内参反投影点云；
5. 如果快照含采集时 EE 位姿并提供手眼标定，同时输出 xArm 基座坐标系点云。

历史 `bbox`、`mask.npy` 和目录标签只用于离线评价；历史掩码不会参与新流水线的候选选择。参考图库是正常的任务先验，不是几何真值。若没有指定药盒的参考图，当前版本只能给出药盒实例，不能可靠确认具体商品身份。

## 环境

本次验证使用独立环境 `/home/wpy/.conda/envs/armcode_recognition_v2`，没有更改原机器人环境。官方 YOLOE 固定在 commit `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`。

关键版本和权重校验值见项目根目录的 `requirements-perception-v2.txt` 和 `PERCEPTION_V2_INITIAL_REPORT.md`。`dependencies/yoloe/`、模型和 `runs/` 均由项目现有 `.gitignore` 忽略。

本次实际采用的隔离安装方式如下；当前机器已经完成，无需重复执行：

```bash
cd /home/wpy/pythonproject/armcode-agent
conda create --prefix /home/wpy/.conda/envs/armcode_recognition_v2 \
  --clone /home/wpy/.conda/envs/perception_env
git clone https://github.com/THU-MIG/yoloe.git dependencies/yoloe
git -C dependencies/yoloe checkout 40cd606cabdbe2b566d6f14a6b162c89206e9a1b
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python -m pip install --upgrade setuptools
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python -m pip install \
  -r requirements-perception-v2.txt
```

权重来源遵循 YOLOE 官方 README：`yoloe-v8s-seg.pt` 来自其 Hugging Face 模型页，`mobileclip_blt.pt` 来自 Apple MobileCLIP 官方下载地址。下载后必须与阶段报告中的 SHA-256 一致。

## 单帧运行

以下命令只读保存数据，不初始化机器人、相机或夹爪服务：

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python -m perception_v2.run_snapshot \
  --snapshot-dir runs/grasp_snapshots/20260520_131624_西瓜霜 \
  --output-dir runs/perception_v2/xiguashuang_v2 \
  --gallery-root runs/grasp_snapshots \
  --warmup \
  --evaluate-reference-mask
```

主要输出：

- `comparison.png`：历史掩码、新语义掩码、深度清理掩码、抓取掩码并排图；
- `semantic_mask_raw.{png,npy}`：完整视觉语义掩码；
- `visible_surface_mask.{png,npy}`：同时具有可靠实测深度的可见表面；
- `grasp_mask.{png,npy}`：去除边缘后的 AnyGrasp 输入掩码；
- `pointcloud_camera.ply`：相机光学坐标系点云，单位米；
- `pointcloud_xarm_base.ply`：经采集位姿和手眼标定变换后的 xArm 基座系点云；
- `perception_report.json`：模型、选择证据、耗时、质量指标及所有产物路径。

点云快速查看：

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python - <<'PY'
import open3d as o3d
p = o3d.io.read_point_cloud('runs/perception_v2/xiguashuang_v2/yoloe_001/pointcloud_xarm_base.ply')
o3d.visualization.draw_geometries([p])
PY
```

## 42 组保存数据开发集对比

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_recognition_v2/bin/python -m perception_v2.benchmark_snapshots \
  --snapshots-root runs/grasp_snapshots \
  --output-dir runs/perception_v2/benchmark_42 \
  --save-visualizations 8
```

这里使用历史框作为 evaluator 参考，以检查新方案是否找回原任务指定物体；它不是人工标注精度测试。身份图库会排除当前被测快照，但同一批数据仍属于开发集，不能据此声称对未知 SKU 的泛化精度。

## 安全边界

- 入口不导入 `robot_service`、夹爪或运动执行接口；
- 不自动下发抓取；AnyGrasp 仍需做碰撞、工作空间和规划可行性检查；
- 基座系点云依赖历史 EE 位姿和 `config/calibration/wrist_handeye.yaml`，只能认为代码链路已核对，不能替代现场标定复验；
- 反光包装的深度空洞不会被虚构填充，因此 RGB 语义掩码可能完整，而点云/抓取掩码保留真实空洞。

## 遮挡推理第一步：导出全场观测证据

`occlusion_reasoning.from_perception` 消费已有报告和原始 RGB-D，无须重新运行
YOLOE / MobileCLIP，也不调用机器人。它读取所有 `instances` 的原始掩码，
而不是只使用 `selected` 的抓取点云：

```bash
.venv-robocasa/bin/python -m occlusion_reasoning.from_perception \
  --report runs/perception_v2/xiguashuang_v2/perception_report.json \
  --snapshot-dir runs/grasp_snapshots/20260520_131624_西瓜霜 \
  --query 西瓜霜 --frame-id real_sample_001 --assume-aligned \
  --output-dir runs/occlusion_input/real_sample_001
```

该历史快照缺少显式对齐元数据，`--assume-aligned` 会在报告中记录假设，不代表
完成了现场标定验证。新采集应保存 `depth_aligned_to_color=true`、
`capture_timestamp_s`，以及采集时刻的相机位姿。明确标记未对齐的深度不能通过
该选项绕过。RGB-D 的尺寸或掩码尺寸不一致会报错，不自动缩放。
点云反投影采用针孔模型，要求图像已经校正或畸变可忽略；适配器不会自行验证
镜头畸变、时间同步或手眼标定。

导出 `scene_observation.json`、`observation_arrays.npz` 和 `instances_overlay.png`。
JSON 中的 `*_key` 对应 NPZ 中的数组名。报告含每个候选的原始可见掩码、有效深度
支持的掩码、相机系点云和可见表面统计；不会执行窄深度带过滤、最大连通域选择或
抓取腐蚀，因此保留细边与不连通的可见片段。深度有效性目前仅检查有限且大于零，
并不等于已处理传感器噪声。可见表面的中心与包围盒不是完整物体位姿与尺寸。

近乎相同的掩码按 IoU 阈值形成重复检测假设，保留所有原始候选及其身份分数，
不把不同类别提示对同一药盒的检测直接算作多个确定物体。节点编号仅在当前帧
有效，`track_id=null`；跨帧关联仍待实现。身份相似度不转换为概率，
`selected` 不等于已确认目标，评价字段不进入推理输入。

`depth_missing` 表示深度缺失；`unassigned_valid_depth` 表示有深度但未被候选
解释的像素，其中可能包含桌面、背景与漏检物体。两者都不是已经推断出的隐藏
物体区域。关系边状态为 `edges_not_inferred`；目标候选包括已观测实例、
未观测/漏检假设与目标不存在假设，概率均为 null，后续由推理模块估计。

第一步还需要采集协议补齐：工作空间范围和支撑面、带时间戳的相机位姿、全场
候选覆盖、历史动作与跨帧实例关联。当前默认 prompts 偏向药盒，不能保证发现
杯子、布块等遮挡物。空候选列表不能据此判定场景为空或目标不存在。

可通过 `--camera-pose FILE.json` 显式提供当前采集对应的坐标变换：

```json
{
  "reference_frame": "robot_base",
  "T_reference_from_camera_optical": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

上面的单位阵仅演示格式，不是实际标定值。平移单位米，方向是光学相机系到公共
参考系。适配器不使用旧快照中的 `current_ee_pose_mrad` 代替采集时位姿。旧报告
没有输入图片哈希，因此报告与快照的对应关系仍依赖调用方；导出记录输入哈希供
后续追踪，但不能倒推证明它们在原推理时已经匹配。

## 遮挡推理第二步：单帧关系证据

在上一步导出观测后运行：

```bash
.venv-robocasa/bin/python -m occlusion_reasoning.extract_relations \
  --observation runs/occlusion_input/real_sample_001/scene_observation.json \
  --output-dir runs/occlusion_relations/real_sample_001
```

入口读取统一观测里的原始可见掩码和对齐深度，不重跑识别模型。核心依赖只有
NumPy、OpenCV、SciPy，均包含在 `requirements-robocasa-scene.txt` 中。
Python 接入方式为：

```python
from occlusion_reasoning.relations import RelationConfig, extract_relations

# packet、arrays 为 build_observation 的两个返回值。
relations = extract_relations(packet, arrays, RelationConfig())
edges = relations["relation_graph"]["edges"]
```

算法不读取目标名称、类别、身份分数或物体位姿，因此可替换上游识别器；结果仍受
实例分割覆盖率、掩码边界和深度质量影响。它先找到相近的实例边界，再比较边界
两侧属于各自实例的实测深度。边界配对要求互为最近距离，避免小物体的外轮廓
跨过可见细边重复配对；等距像素允许并列，不依赖 KD-tree 任意选择的索引。
局部 5×5 中值不使用其他实例或背景像素，不用整物体
深度中位数代替边界深度；共享掩码像素不参与深度投票。配对路径经过第三个实例时
丢弃该配对，避免跨过中间物体建立直接关系。

每对实例保留证据，即使没有生成边：

| 字段 | 含义 |
| --- | --- |
| `mask_overlap` | 原始掩码共享像素和占较小掩码的比例 |
| `boundary` | 最小边界距离、近邻配对数、被第三个实例阻断的数量、近接比例 |
| `depth` | 有效配对数/比例、局部 B−A 深度差中位数及 MAD、前后顺序票数比例 |
| `completion` | 凸轮廓补全缺口及其与前景掩码的交叠，仅为形状先验 |
| `samples` | 最多 48 组可检查的边界坐标和局部深度，缺失深度为 null |
| `edges` | 从这些证据产生的有向关系；方向均为前景 → 后景 |

有两种边：`locally_in_front_of` 只说明采样边界处的前后顺序；
`possible_occludes` 还要求边界接近，且前景进入后景可见掩码凸包内的缺口。
凸形投影先验适合当前药盒基线，不是真实完整轮廓。真实凹形物体、错误分割
可能造成误报；遮挡后仍留下凸轮廓的情形可能漏报。可加 `--no-convex-prior`
只输出局部深度顺序，后续再替换补全模型。

默认近邻半径为 6 px，近接距离 2 px，局部深度差阈值 8 mm，至少 8 个有效配对、
50% 有效深度覆盖和 80% 方向一致性；遮挡假设还要求至少 80% 配对近接、被前景
解释的凸包缺口至少 8 px 且占全部缺口至少 10%。这些是开发用阈值，尚未经过传感器标定或
统计校准。CLI 可调整 `--neighbor-radius-px` 和 `--depth-threshold-m`，
其余阈值通过 `RelationConfig` 设置。像素阈值依赖图像分辨率。

掩码重叠过多、深度不足、深度差过小或局部顺序相互矛盾时保留相应状态，不强制
给出方向。`no_near_boundary_evidence` 表示没有取得近邻边界证据，不能解读为
“确定不存在遮挡”。所有边的 `confirmed=false`、`calibrated_probability=null`；
`evidence_score` 只是方向一致性、有效覆盖等量组合出的启发式分数。

输出为 `relations.json` 和 `relations_overlay.png`。JSON 独立保存关系图、证据和
输入哈希，不修改原始观测；图中 N0 等编号对应原观测节点顺序，紫色点标出边界
样本，橙色箭头标出遮挡假设。默认复用原快照 RGB，也可用 `--rgb` 指定图片；
没有图片时绘制掩码画布。

当前历史 `real_sample_001` 的 3 个实例产生 3 对关系，均没有近邻边界证据，与
图中分离摆放的药盒一致。这只验证数据接入和该负例，不能据此给出真机遮挡准确率。
完全不可见的物体不会被创建成节点；隐藏目标空间分布、支撑关系、跨帧确认和
抓取可行性仍需后续模块处理。

```bash
.venv-robocasa/bin/python -m pytest -q \
  tests/test_occlusion_observation.py tests/test_occlusion_relations.py
```

## 双视角遮挡入口

新增 `occlusion_reasoning.scene.infer_scene`，输入两份同步观测包，输出匿名实例关联、
受遮挡视角、直接投影证据及可选的平面延续候选。完整流程、阈值、仿真评价与局限见
[`robocasa_occlusion/README.md`](../robocasa_occlusion/README.md#双视角遮挡算法观测证据与平面候选分开输出)。
它不读取药名、OCR/检索分数或 `selected` 结果，也不导入硬件服务。

除了各视角的原始可见掩码、对齐深度及内参，还需要在采集快照的 `grasp_input.json`
中记录以下信息。示例值仅展示字段格式，不能给旧照片补写虚构的同步时间：

```json
{
  "camera_id": "left_camera",
  "capture_group_id": "same_static_capture_001",
  "capture_timestamp_s": 1234.5,
  "timestamp_domain": "shared_host_monotonic",
  "depth_aligned_to_color": true
}
```

两个相机的 `camera_id` 不同，采集组、时间域相同；各自采集时的光学坐标系位姿，
通过 `from_perception --camera-pose` 的 JSON 提供。两个位姿必须使用相同参考系。
内参须对应实际对齐后的像素网格；图像需已校正畸变或可近似使用针孔模型。
只提供 RGB 的全局相机暂时不能直接使用这个入口。

`from_perception.export_report` 现已传递上述采集元数据。缺少字段的旧单帧导出仍
可用于原有单帧关系提取，但双视角模块会明确拒绝缺同步、缺位姿或仅假定配准的输入。

```python
from occlusion_reasoning.scene import infer_scene

result = infer_scene([
    (left_observation, left_arrays),
    (wrist_observation, wrist_arrays),
])
# 对非平面/凹形物体，可设置 planar_prior=False。
```

`geometry_health` 检查跨视角深度矛盾，触发时停止关联和直接跨视角推理，仍保留单视角
平面候选。它不能代替标定验收。`occludes_observed_surface` 与 `possible_occludes`
必须区分使用；后者依赖形状先验。两视角都漏分割或完全不可见的物体不会被凭空恢复。
输出实体 ID 只在当前采集组有效，清障执行后应重新观测，不应直接沿用旧关系。

## 真机第一轮：相机采集与单视角验证

新增专用入口 `run_occlusion_camera_test.sh`，默认 Python 为
`/home/seur/data/wpy/conda_envs/env_armcode_agent/bin/python`，可通过
`OCCLUSION_CAMERA_PYTHON` 覆盖。采集过程只构造 `RealSenseDualCameraService`，
读取 `config/hardware/fr3.yaml` 的 `cameras` 字段；不构造机械臂、夹爪、Agent、
ROS 或标定相机的机械臂状态查询封装。采集结束或启动失败时均关闭相机流。

```bash
# 输出目录必须尚不存在；省略 --output-dir 时自动创建时间戳目录。
bash run_occlusion_camera_test.sh capture \
  --output-dir runs/occlusion_real/scene_01

# 只读保存帧做分割与推理，此阶段不连接相机。
bash run_occlusion_camera_test.sh analyze \
  --capture-dir runs/occlusion_real/scene_01 \
  --output-dir runs/occlusion_real/scene_01_analysis
```

此入口给固定和腕部相机**都启用深度**，按配置的 640×480、6 FPS 采集；
由 SDK 将深度对齐到颜色图，保留原始 uint16 深度、米制 `.npy`、内参、帧号、
传感器时间域及主机接收时间。双相机按主机接收时间近似配对，
`passes_host_receive_skew_screen` 仅表示通过接收时差筛查，不等于曝光同步，
也不等于硬件触发同步。适用条件是静态场景。两台设备的硬件时钟不能直接混用。

第一轮分析使用本地 SAM3 权重，通用提示为 `box bottle plate cup`，生成实例
候选及掩码，不执行 OCR 或指定药名绑定。它不是无提示、覆盖所有物体的分割器。
可通过 `analyze --prompts ... --confidence ...` 改变分割前端，推理输入仍是匿名掩码。
先运行原有单帧边界关系，再运行平面补全候选；输出的遮挡均保留假设性质。

没有相机外参时，每个视角使用**自己的光学坐标系**，自变换为单位阵：
`global_optical_local` 和 `wrist_optical_local`。这只是局部平面拟合所需的坐标定义，
不代表已知相机到基座或相机之间的外参。把两份结果送入双视角模块会因参考系不同
被拒绝，不能通过给它们填同一个单位阵来绕过标定。非零镜头畸变也会被拒绝，需先校正。

2026-10-21 已完成一次真实采集：

| 相机 | 序列号 | 整幅图有效深度比例 | 实例候选 | 可能遮挡物体对 |
| --- | --- | --- | --- | --- |
| 固定 | 346522072096 | 92.35% | 9 | 2 |
| 腕部 | 349622072103 | 99.46% | 3 | 1 |

数据为 `runs/occlusion_real/static_first_capture/`，分析为
`runs/occlusion_real/static_first_analysis/`。本次接收时差约 47 ms，仅用于静态采集
记录，未评价曝光同步。两台相机畸变系数均为零，颜色/对齐深度内参一致。
未使用、未补写固定相机的外参，也未把采集时的机器人位姿当作已知信息。

采集目录包含：

```text
capture_report.json              深度质量、串号、近似配对状态
camera_preview.png               两相机颜色 / 深度对照
{global,wrist}/color.png
{global,wrist}/depth.png          原始 uint16 深度
{global,wrist}/depth_m.npy        米制光轴 Z 深度
{global,wrist}/grasp_input.json   本次实际采集元数据，无外部位姿
```

分析目录包含每个视角的 `evidence.png`、`segmentation_report.json`、
`single_view_report.json`、统一 JSON/NPZ 观测包、边界证据与平面候选。
`test_report.json` 汇总两视角结果。叠加图的橙色箭头表示可能遮挡，N 编号只在该帧有效。

本次可视检查发现：固定视角能覆盖主要桌面，但分割候选混入背景物体，且白色小瓶
未被默认提示检出；腕部视角偏向桌沿，三个候选都接触画面边缘，药盒和盘子被截断。
因此输出只算真实数据通路验证，不是人工真值评测，更不能据此宣称恢复所有遮挡。
后续应限定工作区、验收可见掩码，并在保持机械臂不动的情况下人工布置无遮挡、
部分遮挡、完全遮挡和近邻不遮挡四类测试。缺少双相机外参时继续做单视角评价。

可在不连接设备或加载权重的情况下检查清理、单位、配准与禁止伪造外参的合同：

```bash
.venv-robocasa/bin/python -m pytest -q \
  tests/test_occlusion_camera_only.py tests/test_occlusion_multiview.py \
  tests/test_occlusion_observation.py tests/test_occlusion_relations.py
```

本次上述 72 项测试通过。假相机测试会拒绝机械臂/Agent 模块导入；真实采集的两路
流均正常关闭。实际 SAM3 在单独 GPU 进程中读取保存帧，结束后退出。

## 瓶状物召回改进与双相机融合准备

当前 `analyze` 默认采用 `multiscale` 模式，提示为
`box bottle cylindrical container plate cup`，分数阈值仍为 0.35。
全帧预测后，再对四个相互重叠、宽高为原图 65% 的局部窗口预测。
只将局部掩码平移回原像素位置，不填补隐藏区域；接触内部裁剪边缘的残缺预测被剔除，
原始图像边缘的真实截断实例仍保留并标记。掩码 IoU 至少 0.85 才组成重复假设，
采用 complete-link 分组，避免局部重叠链把不同物体连起来。
每组保留一个代表掩码及其原始单次得分，不合并掩码，也不因重复提示或窗口而叠加置信度。

```bash
bash run_occlusion_camera_test.sh analyze \
  --capture-dir runs/occlusion_real/static_first_capture \
  --output-dir runs/occlusion_real/refined_analysis_v1
# 此目录已存在；重跑请选择新的输出目录。
```

可用 `--proposal-mode full` 关闭局部复检；用
`--proposal-mode full --prompts box bottle plate cup` 重现原四提示方案。
`raw_proposal_masks.npz` 与 `segmentation_report.json` 的 `proposal_audit` 保留原始预测、
内部裁剪边缘剔除记录、重复分组及代表来源，便于检查是否误删窄可见部分。

同一张开发帧的对照结果（人工目视近似框，只检查定位，不是独立分割真值评测）：

| 方案 | 固定视角候选数 | 上方白色瓶子 | 盘子右侧长筒 |
| --- | --- | --- | --- |
| 原四提示，全帧 | 9 | 未提出 | 未提出 |
| 增加圆柱容器提示，全帧 | 12 | 0.961 | 0.957 |
| 增加形状提示与局部窗口，去重后 | 12 | 0.965 | 0.957 |

这张图中两个瓶状物的召回改善来自新增形状提示；局部窗口未增加最终实例数。
41 个原始预测中，12 个被判为内部裁剪边缘残缺、17 个归入重复假设，保留 12 个。
腕部保留 4 个实例，但它们仍被原始图像边界截断。原来的背景误检仍在，
没有据此宣称已经得到完整可见实例分割。详情见
`runs/occlusion_real/refined_analysis_v1/comparison.json` 与两视角 `evidence.png`。

双相机几何融合使用固定相机光学坐标作为参考，所需的是
**T_global_optical_from_wrist_optical**，无需先标定机器人基座：

```text
P_global = T_global_from_wrist @ P_wrist
```

可在保持机械臂及两个相机不动时，手动将同一标定板放到两台相机的共视区域，
在多个位置、角度静止采集。已知板的尺寸并分别解算板相对相机的位姿后，每组有：

```text
T_global_from_wrist = T_global_from_board @ inverse(T_wrist_from_board)
```

对多组估计检查一致性，再用未参与求解的板图像检验双向重投影与深度残差。
`relative_pose_from_board` 已实现上述坐标变换组合，但它不检测标定板，也不等于已经
完成实际标定。具体板型、尺寸、原始采样和验证结果仍需在实际标定时记录。
若标定板在两次曝光之间移动，两份板位姿不能直接组合；需在采集时保持板静止。

`config/calibration/static_camera_pair.example.json` 是待测量模板，矩阵为 `null`，
状态为 `pending_measurement`。后续填写实测的相对位姿、正确序列号与测量来源，
保存为独立标定文件，再执行：

```bash
bash run_occlusion_camera_test.sh fuse \
  --analysis-dir runs/occlusion_real/refined_analysis_v1 \
  --calibration config/calibration/static_camera_pair.json \
  --output-dir runs/occlusion_real/fused_scene_01
```

上面的实际标定文件当前尚不存在，此命令仅说明标定完成后的用法。
融合接口会核对米制单位、矩阵方向、刚体约束、设备序列号、原图/深度哈希，
再将两份观测变换到同一参考系。它会使用已有的双视角实例关联、几何矛盾拒绝、
投影深度比较及带视角的遮挡图。分别保留观测支持的关系、平面候选和未知项；
一侧没有实例时，还需区分投影越界、漏分割、缺深度、遮挡和无法确定的自遮挡。

接口只读取保存数据，不连接机器人或相机。相对外参只对当前固定相机对有效：
机械臂带着腕部相机运动后，必须用采集时的手眼/机器人位姿更新关系或重新标定。
目前没有把真实的两个视角合并成一个已标定场景；模板与缺外参输入都会被拒绝。

```bash
.venv-robocasa/bin/python -m pytest -q \
  tests/test_occlusion_proposal_refinement.py tests/test_occlusion_real_fusion.py \
  tests/test_occlusion_camera_only.py tests/test_occlusion_multiview.py \
  tests/test_occlusion_observation.py tests/test_occlusion_relations.py
```

上述 91 项测试已通过，覆盖原像素映射、内部裁剪边缘、重复链、零候选、
相对位姿组合方向、错误串号、未测量外参拒绝、观测包不变性以及离线融合导出。
