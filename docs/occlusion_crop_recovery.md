# 小遮挡物跨裁剪边缘时的局部分割恢复

2026-10-23：已接入原腕部遮挡分析，原 Agent 启动命令不变，退出旧进程并重启即可加载。

此前，多尺度 SAM3 检测到了压在西瓜霜上的小方块，但候选触及局部窗口的右边界 `x=416`，被作为不完整片段剔除。其他窗口没有补出完整方块，因此关系图里只有药盒，没有方块节点和遮挡边。

现在的流程：

1. 全图和重叠窗口直接使用 `box`、`bottle`、`cylindrical container`、`plate`、`cup`、`cube`、`block` 提取候选；小方块检测不依赖先找到触边片段。
2. 触边片段若已被其他完整候选覆盖，直接复用完整候选。
3. 对仍未覆盖的片段，以其可见区域为中心扩展裁剪，用原类别加 `cube`、`block` 重新分割。
4. 将新预测掩码映射回原始图像，走现有去重、深度关系分析和清障选择。

每个片段最多尝试两种窗口大小，每帧最多追加四个窗口，成功补回后立即停止该片段的重试。没有新增 Agent 开关，也没有新增身份、深度覆盖率或清障资格门槛。掩码维度只在映射入口检查一次；模型错误直接上报，不增加宽泛异常兜底。原来的原始 RGB-D、场景有效性、故障停止和执行检查继续沿用。

新掩码来自 SAM3 对完整局部图像的实际预测，不将旧片段拼接成完整物体，不填造深度，不提高重复预测的置信度。日志增加 `recovery_windows`、`recovered_proposals`；每次追加窗口和新掩码记录 `recovery_from_raw_id`，可追溯到旧片段。

## 本次真实照片离线回放

输入为 `runs/current_capture/occlusion_graph/3be2ebcb5cc74abd9a0ef9fcf0a7b353/` 中的原始 RGB-D。未连接相机、机械臂或夹爪。

| 项目 | 原结果 | 更新后 |
| --- | --- | --- |
| 物体节点 | 6 | 7 |
| 小方块 | 缺失 | N6，框 `[378,204,420,251]` |
| 可能遮挡关系 | 0 | N6 → N3（小方块 → 药盒） |
| 清障选择 | `no_search_evidence` | `exploratory_clear_candidate`，选择 N6 |

本次只追加一次局部分割，窗口 `[331,157,463,298]`，耗时约 0.239 秒；整个候选提取约 2.620 秒，不含模型加载。小方块代表预测置信度为 0.926；两处平面延拓证据支持同一个可能遮挡关系，其中一处深度差中位数约 27.6 mm。置信度和几何证据均不等同于抓取成功率。

结果目录：`runs/occlusion_crop_recovery/20261023-b9c54feb29/`。

- `evidence/evidence.png`：新分割结果及 N6 → N3 箭头。
- `evidence/segmentation_report.json`：原片段、新窗口与预测掩码的完整来源记录。
- `report.json`：完整关系证据。
- `summary.json`：回放结果摘要。

本次保持目标身份为未确认，仍由原清障选择器选出方块，证明这一步不依赖先把被遮挡药盒识别成功。实际抓取位姿是否可用、物体是否搬走、移开后能否识别目标，仍由原抓取规划与新观测处理；本次未进行真机动作。

## 00:29 新观测：补上没有触边线索时的入口

最新失败帧 `f052e86ec79b4a3e8af34027c881226e` 和 `fe605321646846468f3f74f6f54670b5` 已加载上一轮更新，报告中有 `max_recovery_windows=4`，但 `recovery_windows=0`。两帧的通用 `box` 提示均未输出小方块候选，因此没有片段可以触发扩窗。上一轮旧图回放只验证了触边恢复，没有覆盖这个完全漏检的入口。

本次将 `cube`、`block` 加入常规全图和重叠窗口的默认提示，而非仅在扩窗时使用。无需新开关，也未改变身份相似度门槛或执行检查。

三帧真实保存图像使用同一组更新后提示进行回放，均得到小方块 N6 → 药盒 N3，清障选择器均选择 N6，无需追加扩窗：

| 输入帧 | 小方块代表预测分数 | 结果 |
| --- | --- | --- |
| 最新补拍 `fe605…` | 0.813 | `exploratory_clear_candidate` |
| 最新首拍 `f052…` | 0.789 | `exploratory_clear_candidate` |
| 前一次失败帧 `3be2…` | 0.820 | `exploratory_clear_candidate` |

完整证据：`runs/occlusion_crop_recovery/20261023-direct-prompts-64ffa003/`。这些分数是代表掩码的模型分数，不是物体身份或抓取成功率。

另使用原 `WristOcclusionService.analyze()` 和原恢复动作菜单检查两种真实保存场景：最新遮挡帧产生 `clear:…/instance_006`；方块放在药盒旁边的 `10446…` 帧没有遮挡边，也没有清障动作。证据位于 `runs/occlusion_crop_recovery/20261023-agent-prompts-ec08cbdd/`，包含实际服务生成的掩码、关系图和动作菜单。没有执行移动。

## 验证

```bash
.venv-mobileclip2-fr3/bin/python scripts/check_core.py
```

核心回归：481 项通过、2 项跳过。新增覆盖触边片段重新分割、未知目标下形成遮挡边并选择清障物、已有完整掩码时省去重复推理、补分割失败时不把旧片段当作完整物体，以及没有任何通用 `box` 线索时仍通过默认 `cube` 提示检出方块。

可以用已有离线入口重新处理保存的帧，输出目录需为新目录：

```bash
.venv-mobileclip2-fr3/bin/python -m occlusion_reasoning.real_single_view \
  --capture-dir runs/current_capture/occlusion_graph/3be2ebcb5cc74abd9a0ef9fcf0a7b353 \
  --output-dir runs/occlusion_crop_recovery/replay_new \
  --checkpoint /home/seur/data/wpy/models/sam3/sam3.pt \
  --cameras wrist
```
