# 待统一集成的接口清单

本阶段只增加仿真模块、配置、依赖与测试。下列生产文件均未修改。
不需要修改 CaP 生成语法或 `runtime.cap_policy.PolicyInterpreter` 的受限 API：
策略仍然只调用 `observe(name) -> bool`、`pick_place(name, destination) -> bool`。
Backend 的 Python API 返回 `SkillResult`，独立 `policy.py` 的闭包负责转换为 bool，
验证有序任务对、观测预算及失败停止；该示例没有替代完整生产调度器。

| 集成位置 | 当前约束 | 需要统一决定的变更 |
| --- | --- | --- |
| `runtime/config.py`、`app/cli.py` | 策略类型与硬件执行配置混在现有入口中 | 新增执行 backend 选择，独立于 `policy_backend=cap/hybrid`；默认保持原行为。增加仿真配置、种子、独立输出路径 |
| `runtime/services.py`、`agents/simple_execution_runtime.py` | 构建硬件/相机 context，再运行动作列表 | 仿真分支应在构建硬件服务之前选定；注入 Backend 工厂和生命周期，不创建 FR3、RealSense、串口、ROS 或 VLM 服务 |
| `agents/cap_execution.py::run_policy` | observe 直接调用 robot；pick_place 闭包调用 `_run_actions` 执行 grasp/place 列表 | 抽取或注入受限技能绑定，保留 alias 绑定、任务顺序、重复动作检查、观测预算及失败报告语义 |
| `agents/hybrid_execution.py::HybridTaskRunner.execute_node` | 每节点执行闭包绑定 `_run_actions` 和硬件恢复状态 | 注入同一技能绑定，同时保留节点 revision、完成前缀、预生成作废、动作只执行一次的检查；本阶段不改双流水线 |
| `core/skill_base.py::SkillResult` | success/data/message/error 已足够 | **无需改变类型**。约定标准化 observation 和 execution evidence 的 data 字段；数组需经 artifact 引用或序列化传输 |
| `core/context.py`、相机/观测缓存 | 真实观测含相机 epoch、捕获位姿、身份与抓取证据 | 为仿真定义 `episode + step + frame=sim_world` 版本绑定；明确深度单位、光学变换和 `simulator_ground_truth`。不要把真值伪装成识别结果，不沿用旧硬件观测 |
| 任务/容器解析及 alias | 生产使用药盒、盘子、动态 reference ID | 在适配入口显式映射名称到仿真实体和目标区；当前只支持 red_cube/blue_cube 与 red_goal/blue_goal。目标区不是生产容器检测结果 |
| `runtime/task_state.py`、失败报告 | 依赖生产动作完成列表、holding、recovery route | 把仿真技能阶段、抬升/接触/放置证据转换成统一结果；明确复合技能与 grasp/place 两条完成记录的对应关系。失败不得把未完成物体标记完成 |
| `runtime/timing.py` 与性能报表 | 生产 spans 使用墙钟时间 | 增加明确 backend/time_domain；保留仿真步数、sim_duration_s 与 wall_duration_s 两列，禁止用仿真速度推断真实机器人吞吐 |
| 进程和线程边界 | FR3 依赖与仿真依赖存在版本差异 | 决定使用独立进程 RPC 或专用启动解释器。MuJoCo/EGL 实例归属一个线程，策略后台预生成线程不可调用仿真对象 |

推荐先定义独立技能绑定 Protocol/工厂，并将现有闭包改为依赖注入；这需要统一集成时
审查公共接口。本次没有为了“可插拔”而提前修改生产文件，也没有把 Backend 冒充为
拥有全部 grasp/place/home/恢复能力的 SimpleRobotActions。

验收统一集成时至少应检查：两个 CaP/Hybrid 节点能执行；一次失败不会重放完成节点；
reset 后旧观测不可复用；无硬件服务创建；并行预生成不并行操作仿真；日志明确区分
时间域。继续沿用现有遮挡默认配置，视觉接入与真值使用范围需单独约定。
