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
