# PMIR + robosuite 初步执行实验

从受限 CaP Python 自动生成微操作依赖图，在已有 Panda/MuJoCo 接触物理后端上执行。
本目录是独立实验入口，不接入 FR3、不读取真实相机或历史标定，不改变已有 Agent 默认行为。

```python
def policy():
    pick_place("red_cube", "red_goal")
    pick_place("blue_cube", "blue_goal")
```

每个抓放展开为 `CAPTURE → PERCEIVE → GRASP_PLAN → MOTION_PLAN`，随后是抬高、接近、
下降、关爪、抬升、运输、下降、张爪、撤离和稳定验证。物理操作保持程序顺序；后续独立计算
可以提前执行。读写关系来自技能契约，分析器为具体程序自动实例化依赖，未实现从任意技能
代码自动推断物理影响范围。

- `runtime/pmir/graph.py`：节点、带类型和原因的依赖边、DAG 校验、JSON/Mermaid 导出。
- `runtime/pmir/compiler.py`：受限 AST、技能注册表、数据/控制/物理读写依赖。
- `runtime/pmir/scheduler.py`：资源容量约束、就绪调度、事件放行、分支跳过、失败停止。
- `adapter.py`：单线程 MuJoCo/EGL、逐控制步协作式微操作、接触与放置验证。
- `perception.py`：只读取冻结 RGB-D 的已知颜色/尺寸方块几何估计。
- `experiment.py`：相同种子、技能、资源额度、成功判据的串行/动态图对照。

## 运行

在活动仓库根目录使用已经安装的仿真环境；不需要重新安装依赖：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv-robosuite/bin/python -m simulation.pmir.experiment \
  --output-dir runs/experiments/pmir_new_native \
  --seeds 7 8 9 --warmups 1
```

输出目录必须尚不存在。默认包含 `independent` 和 `occluded` 两种场景、`serial` 和
`dynamic` 两种执行模式。测量前每种场景/模式以第一个种子预热一次；各轮轮换模式顺序，并打乱场景块顺序。
独立场景使用种子扰动方块初始位置；遮挡场景固定对齐布局，多个种子只是重复运行，不能当成
多种遮挡布局的泛化测试。

明确注入感知延时的调度实验：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv-robosuite/bin/python -m simulation.pmir.experiment \
  --output-dir runs/experiments/pmir_new_latency \
  --seeds 7 8 9 --compute-delay-s .2 --warmups 1
```

仍然运行实际 RGB-D 处理和实际物理抓放；额外的 `.2` 秒是人为延时，不能称为模型推理时间。
不注入延时的结果必须单列；未测得收益时保留该结论。

让仿真控制步按 20 Hz 墙钟节奏运行，并在等待计算时继续保持控制：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv-robosuite/bin/python -m simulation.pmir.experiment \
  --output-dir runs/experiments/pmir_new_paced \
  --scenarios occluded --seeds 7 --warmups 0 \
  --compute-delay-s .2 --clock real_time
```

`real_time` 是尽力维持节拍，报告包含迟到次数与最大迟到，不是硬实时保证。
默认 `fast` 模式在等待纯计算时暂停物理步进；其墙钟收益表示主机仿真实验的完成时间变化。
`sim_duration_s` 是仿真控制时间，两者都不能直接解释为真实 FR3 吞吐。

## 可见事件与验证

遮挡场景只改变仿真固定相机的布局，初始蓝方块有效颜色/深度掩膜必须少于 25 像素。
实际抬升、释放和稳定放置完成后，主线程拍摄 RGB-D 并检查目标掩膜和仿真物体速度。
只有掩膜至少 25 像素且物体稳定，才产生 `visible:<target>` 事件。事件实际发生的微操作
被写入动态依赖图；没有按预计动作时长百分比或逻辑计时器释放依赖。

感知和解析抓取只使用图像、深度、仿真相机标定及已知方块/桌面几何，后台输入不含物体真值。
状态版本、稳定性、双指接触、抬升高度、物体相对夹爪偏移和最终放置判定明确使用仿真真值。
这是受控感知/执行基线，不是通用检测模型、AnyGrasp 或完整碰撞轨迹规划器。

每个控制步监视所有已知物体的位姿变化；位移超过 1.5 mm 或转动超过 0.04 rad 时递增版本。
计算提交及第一次物理动作前检查目标版本。陈旧证据和物理失败停止本轮，保留完成前缀，
并等待后台计算结束。阈值是实验设置，未证明对真实场景足够保守。尚未实现自动局部恢复、
通用区域影响估计、PEIP、跨任务缓存或物理回滚。

## AST 范围与扩展

支持无参数 `policy()` 或顶层技能调用、常量/局部字符串绑定、至多 16 项的字面量循环、
`if observe(name)` / `if not observe(name)` 及其 else。分支内部工作必须等待真实条件结果，
不预测分支。显式 `observe` 保留程序位置；只有技能内部的无副作用准备阶段可以提前运行。

不支持任意 Python、动态 range、helper 递归、属性访问、导入或分支内部提前 return；
包括既有 CaP 的某些 fail-fast 写法会显式拒绝。没有修改生产 AST 解释器或声称支持全部 CaP。

目前有展开描述的技能是 `observe` 和 `pick_place`。`SkillRegistry.register_blackbox`
可为新增技能生成保守的全状态屏障，保留完整参数；仿真入口在没有专门 handler 时拒绝执行。
开抽屉、倒水的微操作与物理实现尚未加入。

## 输出与核对

每轮保存策略、初末 PMIR 图、全部事件、逐节点时间区间、物理后端日志、初末图像和
可见性检查时的 RGB-D。根目录保存运行顺序、代码哈希、参数、软件版本与配对统计。
只有初始 qpos/RGB 哈希相同、两轮均成功、物理动作顺序一致且 fast 模式控制步数相同时才计算速度比。

`wall_duration_s` 包含执行期间的图像采集、可见性检查、版本监测、调度、验证、事件日志和
后台收尾；编译、环境初始化、结果图像导出另外记录或排除。`spans` 的物理条带是微操作占用
区间，可能包含主线程穿插采集；另报计算与控制步调用区间（含后端状态检查）的交集，不能混称。

使用已有感知环境导出可分享的图，不往仿真环境安装绘图库：

```bash
.venv-mobileclip2-fr3/bin/python scripts/plot_pmir.py \
  --run-dir runs/experiments/pmir_new_latency \
  --output-dir runs/experiments/pmir_new_figures
```

图中为可读性移除传递冗余边；完整 JSON 仍保留所有依赖原因。

```bash
.venv-robosuite/bin/python -m pytest -q tests/test_pmir.py tests/test_pmir_robosuite.py
```

本轮具体数据与局限见 `docs/pmir_robosuite_20261110.txt`。

## 计算延迟控制实验与耗时拆分（2026-11-10 补充）

在同一个现有入口运行三种模式，无新增规划、预测或恢复模块：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv-robosuite/bin/python -m simulation.pmir.experiment \
  --output-dir runs/experiments/pmir_latency_sweep_new \
  --seeds 7 --repeats 6 --warmups 1 \
  --modes serial naive dynamic --delays-ms 0 50 100 200 500
```

- `serial`：微操作依次执行。
- `naive`：同一静态依赖图、资源额度、物理动作顺序与版本校验，但首次隐式采图忽略
  可见性门控。若该帧的目标不可见，丢弃结果，确认可见且稳定后只允许重新采图计算一次。
  显式 `observe` 的分支语义不改。这是可审计的对照定义，不声称代表所有普通流水线。
- `dynamic`：首次隐式采图就等待实际可见/稳定事件，避免上述无效早算。

每个延迟是**每次 PERCEIVE 的额外等待**，包含失败的早算；两次有效感知共需 2 倍该延迟，
Naive 若发生一次重算则需 3 倍。真实 RGB-D 几何计算仍照常执行。该 sleep 注入衡量延迟隐藏，
不模拟 CPU/GPU 饱和、GIL 竞争或真实模型吞吐。

固定 seed 7 分别固定两种场景的布局；场景之间相机/遮挡条件不同。每种场景/延迟块中三种
方法相邻运行，六轮遍历三种模式的六种排列，每种方法在每个位置出现两次。每轮打乱各块
顺序。预热在最大注入延迟运行，每种场景/方法一次，排除于正式结果。该设计仅检验无遮挡
和初始完全遮挡两个条件，不是连续遮挡程度扫描。生产默认策略与硬件配置不变。

新增 `cost_profile.json` 保存完整互斥前台区间与等待原因；`result.json` 保存分类总量。
前台总量严格加和等于 `wall_duration_s`，后台计算是另一条可能重叠的时间轴，不能再相加：

| 字段 | 口径 |
| --- | --- |
| `image_acquire_format` | 后端渲染、RGB-D 转换及其内部复制，扣除被单独记录的验证和日志 |
| `image_snapshot_copy` | 额外制作不可变工作线程快照的复制；不代表所有图像复制 |
| `state_validation` | 可见性掩膜、速度/版本、接触和放置状态检查；掩膜处理归在这里避免重复统计 |
| `scheduler_decision` | 依赖/资源/分支/就绪扫描和提交工作分派 |
| `physics_step` | MuJoCo 物理积分及控制器执行，包括夹爪和稳定保持；不是纯真实机械臂运动 |
| `robot_control` | 微操作推进、轨迹插值和控制步外围工作 |
| `compute_wait_poll` | 无活跃物理微操作时等待计算的轮询；可能包含计算完成后的尾部 |
| `profiling_analysis` | 判断等待原因的仪表分析开销，单独报告 |
| `logging` / `runtime_bookkeeping` | 写日志及其他前台工作，包括计时边界管理 |
| `perception_wall_s` | 后台 PERCEIVE 总墙钟，包含注入等待；额外报告扣除注入后的墙钟和线程 CPU 时间 |
| `exposed_critical_compute_s` | 无物理操作推进且下一物理操作的数据/资源链被正在运行的计算阻塞的交集 |

最后一项是可直接测得的关键等待链上**暴露的计算时间下界**，排除完成后的轮询尾部，
不是整张执行图的反事实最长路径，也不包括计算拖慢物理积分等间接资源干扰。

三个数必须分开：
1. `compute_physical_residency_overlap_wall_s`：与物理微操作整个占用区间的交集，较宽口径。
2. `compute_actual_physics_overlap_s`：与独占的实际 `env.step` 区间的交集，排除穿插拍照等。
3. `comparison.json` 的 `saved_wall_s`：同初态、同动作/步数、均成功的 Serial 减候选模式。
   这是实测净节省，可能为负；前两项均不能代替它。

`task_timing_wall_s` 还包括调度器构造、最终目标检查和 task_timing 包装，作为总入口复核；
`wall_duration_s` 的互斥分类从调度器开始记录到后台收尾结束，包含初始可见性检查。
环境初始化、编译和运行前后导出图像不在本次执行窗口内。初始 RGB/qpos 哈希、动作顺序、
控制步数、源码副本、软件版本、运行顺序、成功结果均保存在实验目录。不同步数的 fast
配对拒绝计算速度比。错误测量仍保留并计入成功率，不静默删除。

```bash
.venv-mobileclip2-fr3/bin/python scripts/plot_pmir.py \
  --run-dir runs/experiments/pmir_latency_sweep_new \
  --output-dir runs/experiments/pmir_latency_sweep_figures \
  --sweep --delay-ms 200 --scenario occluded --repeat 0
```

导出全扫描曲线、互斥耗时堆积图、无效计算曲线、三模式时间线和实际可见性图像。
只读已有文件即可绘图，不会再次执行仿真或机器人动作。6 次配对的 bootstrap 区间仅作
探索性不确定性指标；盈亏平衡点应给出已测试点或区间，不能声称普适的精确阈值。
