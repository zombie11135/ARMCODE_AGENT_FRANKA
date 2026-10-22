# 模型、SDK 与外部运行环境

本仓库分发 Agent、适配服务和开发工具源码。权重、SDK 授权、第三方源码 checkout、
图库、实机标定和 ROS 工作区由使用者在本地准备。保留 `services/anygrasp_*`、
`services/sam3_*` 等适配代码是为了维持完整接口，它们不是模型权重或外部 SDK 本体。

| 功能 | 本仓库保留 | 本地准备与配置 |
| --- | --- | --- |
| AnyGrasp | FR3 服务、worker、离线工具和技能调用 | SDK、权重、授权和独立 Python；配置 `perception.anygrasp_*` |
| GSNet / GraspNet | 服务和抓取工具 | 对应源码、CUDA 扩展和权重；配置 `perception.gsnet_root` |
| SAM3 | bbox 分割适配服务 | 相容的 Python 包、依赖和模型；配置 `perception.sam3_model_dir` |
| GroundingDINO | 检测适配服务 | 模型目录及 Transformers / Torch 环境；配置 `perception.grounding_dino_model` |
| YOLOE / MobileCLIP | 离线感知和参考图库服务 | 第三方源码、权重与图库；见下文和 references 模板 |
| MobileCLIP2-B | 正式观测技能的图库匹配编码器 | OpenCLIP 3.3.0、timm 1.0.30、本地权重及图库；见 `requirements-reference-recognition.txt` |
| PaddleOCR | 延迟加载和 OCR 服务 | 在对应感知环境配置 PaddleOCR / Paddle 和模型 |
| AnyPlace | 数据准备、几何与推理适配 | 官方源码、权重和独立环境；见 `anyplace_real/README.md` |
| FR3 / Robotiq / RealSense | 硬件接口、坐标变换和工具 | 外部 ROS / MoveIt 工作区、串口及相机依赖、设备和本机标定 |

当前观测流程的 MobileCLIP2-B、GroundingDINO 权重及预建特征库通过 Release 附件分发，
下载、校验和许可证说明见 [观测资源包](docs/observe_assets.md)。

[AnyGrasp 官方授权流程](https://github.com/graspnet/anygrasp_sdk/blob/main/license_registration/README.md)
要求按其流程申请授权。不要提交 `licenseCfg.json`、`.public_key`、`.signature` 或 `.lic`。
[SAM3 官方安装说明](https://github.com/facebookresearch/sam3/blob/main/README.md)
说明了模型获取与安装要求，权重访问及使用遵循其对应条款。
其他依赖同样按各自来源取得，仓库不代为授予使用或再分发权限。

## 按模块建立环境

- `requirements-dev.txt`：本次发布的 Python 3.10 CPU 开发和 CI 环境。
- `requirements-hardware.txt`：硬件适配的部分 Python 依赖，不含 ROS、模型和相机完整部署。
- `requirements-gate-sim.txt`：原 Python 3.9 PyBullet 仿真环境。
- `requirements-gate-learning.txt`：原学习环境版本记录，需要额外准备相容的 CUDA 和模型。
- `requirements-perception-v2.txt`：原感知环境记录；含本地 editable 源码路径，不能在未准备源码时直接安装。
- `requirements-reference-recognition.txt`：正式图库识别使用 MobileCLIP2-B，通过 OpenCLIP 加载本地权重，不需要第一代 MobileCLIP 源码。安装在同时运行 GroundingDINO 的感知环境中；具体配置与离线验收见 [观测技能迁移记录](docs/observe_mobileclip2_20261022.txt)。第一代模型仅保留为旧配置兼容选项。

感知环境原来使用 YOLOE commit `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`。
可在项目根目录准备该 checkout，再按照其官方说明建立独立环境：

```bash
git clone https://github.com/THU-MIG/yoloe.git dependencies/yoloe
git -C dependencies/yoloe checkout 40cd606cabdbe2b566d6f14a6b162c89206e9a1b
```

`dependencies/` 被忽略且不会被源码导出器打包；这不是 Git submodule。
不同环境的 NumPy、Torch、OpenCV 等约束可能冲突，不要合并全部 requirements。
本次发布验证范围是 CPU 基础回归，不代表在新机器重新安装并验证了全部可选模型。

当前未替项目选定开源许可证。首次多人协作建议使用私有仓库；决定公开及授权方式前，
由项目所有者确认自有代码和既有来源的权利，再添加适当的 LICENSE 与归属说明。
