# 观测模型权重与特征库

[下载资源包：observe-assets-20261022](https://github.com/zombie11135/ARMCODE_AGENT_FRANKA/releases/tag/observe-assets-20261022)

资源包对应 `feat/observe-mobileclip2` 的观测实现（代码提交 `7e5318d`）。
权重与特征库作为 GitHub Release 附件分发；普通 `git clone` 不会下载这些附件。

| 附件 | 压缩后大小 | 解压位置与内容 |
| --- | ---: | --- |
| `mobileclip2-b.tar.gz` | 557.5 MB | `models/mobileclip2-b/`：权重、配置、模型说明、LICENSE、NOTICE |
| `grounding-dino-tiny.tar.gz` | 640.7 MB | `models/grounding-dino-tiny/`：权重、配置、分词器、预处理配置、LICENSE |
| `mobileclip2-gallery.tar.gz` | 101.4 KB | `data/object_gallery/vector_libraries/mobileclip2_b/`：`index.npz`、`metadata.json` |
| `manifest.json`、`SHA256SUMS` | 约 4.4 KB | 文件与压缩包的 SHA256、大小、模型来源 |

大小使用十进制单位。两份权重保持上游原始内容，未重新训练或修改。
第一代 MobileCLIP 对照模型不在正式资源包中。

## 下载和解压

先切换到 `feat/observe-mobileclip2` 或包含该功能的后续代码版本。
在项目根目录执行：

```bash
mkdir -p releases/observe-assets-20261022
for asset in mobileclip2-b.tar.gz grounding-dino-tiny.tar.gz mobileclip2-gallery.tar.gz manifest.json SHA256SUMS; do
  curl -fL --retry 3 \
    -o "releases/observe-assets-20261022/$asset" \
    "https://github.com/zombie11135/ARMCODE_AGENT_FRANKA/releases/download/observe-assets-20261022/$asset"
done
(cd releases/observe-assets-20261022 && sha256sum -c SHA256SUMS)
```

校验全部通过后，在项目根目录解压。已有同路径文件时先核对校验值或备份：

```bash
tar -xzf releases/observe-assets-20261022/mobileclip2-b.tar.gz
tar -xzf releases/observe-assets-20261022/grounding-dino-tiny.tar.gz
tar -xzf releases/observe-assets-20261022/mobileclip2-gallery.tar.gz
```

参考照片与 `data/object_gallery/gallery.json` 已在代码仓库中。
特征库包含四类药品的 45 个 512 维向量：布洛芬 20 张、西瓜霜 12 张、
蒙脱石散 7 张、抗病毒口服液 6 张。正式加载器按 `object_id` 和图片 SHA256
对应特征，兼容仓库的扁平图片目录及本地按类别整理的目录。
`metadata.json` 和向量文件内的路径保留建库时记录，不要求其他机器使用相同目录结构。

## 使用

在同时运行 GroundingDINO 的感知环境中安装 `requirements-reference-recognition.txt`。
新机器可将 `config/perception/references.example.yaml` 复制为
`config/perception/references.yaml`，再按设备修改 `device`；已有本机配置时逐项核对。
硬件 profile 的 `perception.reference_config` 指向该配置，
`perception.grounding_dino_model` 指向本机 `models/grounding-dino-tiny` 的绝对路径，随后重启 Agent。

配置细节、离线回放和验证范围见 [观测技能修改记录](observe_mobileclip2_20261022.txt)。
资源包不包含设备配置、标定、AnyGrasp 权重或 Python 环境；尚未完成真实机器人端到端验收。

## 来源与校验

- MobileCLIP2-B：[Apple 上游模型](https://huggingface.co/apple/MobileCLIP2-B)。
  Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.
  许可证及署名已随包附带，具体使用范围以 LICENSE 为准。
- GroundingDINO tiny：[IDEA-Research 上游模型](https://huggingface.co/IDEA-Research/grounding-dino-tiny)，
  随包附带 Apache 2.0 许可证。
- [资源清单](observe_assets_20261022.json) 固定本轮权重、向量和压缩包的 SHA256。
  发布前已逐文件校验压缩包内容，并用正式加载器确认向量库与共享图库对应。
