# 多人协作约定

采用单仓库、模块分工和 Pull Request。
`feat/occlusion-observation-scheduling` 是当前共享开发分支，接收观测、感知和集成更新；
`main` 是稳定分支，开发分支通过离线检查、相关模型验证和所需实机验收后，
再单独创建以 `main` 为目标的稳定更新 PR，不自动合并或直接推送 `main`。
本机配置、模型、实际标定和运行数据独立管理。实机验收通过的提交打
`robot-validated-*` 标签，并记录验证设备和范围。

## 第一次参与

按 [SETUP](docs/SETUP.md) 建立 CPU 开发环境，运行 `python scripts/check_core.py`。
随后从最新开发分支开个人任务分支：

```bash
git switch feat/occlusion-observation-scheduling
git pull --ff-only origin feat/occlusion-observation-scheduling
git switch -c feat/your-module-your-task
# 完成一个范围明确的任务
python scripts/check_core.py
git add <本次修改的文件>
git commit -m "Describe the behavior change"
git push -u origin HEAD
```

在 GitHub 创建以 `feat/occlusion-observation-scheduling` 为目标的 PR，
说明行为变化和实际验证范围，由另一位开发者评审后合并。
经维护者协调，也可以直接将已验证的小改动推送到共享开发分支；推送前先同步远端。
多人可以同时修改不同模块；同一目录内也按任务开个人分支，完成后合回共享开发分支。
合并方式建议统一为 squash，便于按任务回退。

## 分工与评审

| 工作 | 目录 | 评审重点 |
| --- | --- | --- |
| Agent 规划 | `agents/` | 任务图、动作约定、重规划和失败恢复 |
| 感知识别 | `perception_v2/`、相关 `services/` | 保存场景上的输入输出、坐标系、模型版本 |
| Gate 与学习 | `nesyro_gate/`、`gate_learning/`、`gate_sim/` | 规则、训练数据、决策与离线评估 |
| 技能抓放 | `skill_lib/`、`anyplace_real/` | 缓存和状态、动作顺序、几何与失败处理 |
| 系统集成 | `app/`、`runtime/`、`hardware/`、`config/` | 服务装配、设备生命周期、配置和实机验收 |

`core/` 公共接口、`runtime/` 装配、坐标系/单位以及运动链路变更，应由集成人员和
受影响模块的开发者共同评审。拆文件与改变行为尽量分成独立 PR。
核心代码可以在各开发者电脑上修改；实机标定、设备配置和最终运动验收在指定集成电脑完成。
同一机械臂一次只由一个人运行控制任务。

## 验证按任务逐步补齐

基础 CI：`python scripts/check_core.py`，无模型、无设备；开发分支推送和 PR 均运行检查。
观测模块入口、职责和验证边界见 [观测开发指南](docs/OBSERVATION_DEVELOPMENT.md)。
感知/学习：在独立环境使用保存数据或合成样例，PR 记录数据标识、依赖版本和命令。
硬件：记录 commit、设备 profile/标定版本和结果；不能验证时明确写「尚未实机验证」。
新样例优先使用小型合成数据；`data/object_gallery/` 的清单、参考图片和采集记录
随仓库共享，特征缓存、运行锁、其他采集数据和运行目录不提交。
新增可移植测试先在干净目录验证，再加入 `scripts/check_core.py`。

仓库管理员在首次 push 后设置 `main`：要求 PR、至少一人评审、`core / offline` 检查通过，
并限制直接推送和 force push。实际可用规则以仓库设置为准；这些服务端规则不会因添加 YAML 自动生效。
暂未填入具体 GitHub 用户的 CODEOWNERS，团队成员确定后再配置模块负责人。
