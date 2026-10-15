# 简短技能使用文档：observe / grasp / place

本文档描述当前 `--simple_actions` 链路中的三个简短动作接口：

```python
observe(obj)
grasp(obj)
place(container)
```

这三个函数定义在 `skill_lib/simple_actions.py` 中。它们本身只生成 `SimpleAction`：

```python
def observe(obj: str) -> SimpleAction
def grasp(obj: str) -> SimpleAction
def place(container: str) -> SimpleAction
```

真正执行动作的是 `SimpleRobotActions` 和 `SimpleExecutionRuntime`。

## 总体流程

在 `--simple_actions` 模式下，整体链路是：

```text
用户任务
  -> PlanningAgent 生成 PlanGraph
  -> SimpleCodeGenerator 生成 observe/grasp/place 动作列表
  -> SimpleExecutionRuntime 同进程执行动作
```

生成的 `runs/generated_code/current_plan_executor.py` 主要用于查看动作，不是主要执行入口。真实执行发生在：

```text
agents/simple_execution_runtime.py
```

## observe(obj)

### 用途

`observe(obj)` 是辅助观察动作，用于让机械臂移动到更适合观察的位置，并尝试缓存目标信息。

它不会抓取，也不会放置。

### 参数

```python
observe("布洛芬")
observe("蒙脱石散")
observe("container")
observe("粉色盘子")
```

`obj` 可以是药盒名称，也可以是容器名称。

### 药盒 observe

如果 `obj` 看起来不是容器，则按药盒处理：

```text
如果已有该药盒缓存 bbox / grasp_input
  -> 根据缓存 bbox 和深度估计目标上方观察位姿
  -> 移动到 z=450
  -> 重新拍 wrist 图
  -> 重新 OCR/识别药盒

如果没有缓存
  -> 从 home、home 左 15cm、home 右 15cm 扫描
  -> 找到后缓存 selected_instance 和 grasp_input
```

当前 observe 高度：

```text
observe_z_mm = 450
```

注意：在 `SimpleExecutionRuntime` 里，`observe` 失败是软失败。也就是说，如果 `observe("西瓜霜")` 没有稳定识别到目标，它会打印 warning，但不会中断任务，后面的 `grasp("西瓜霜")` 仍会继续执行。

### 容器 observe

如果 `obj` 看起来像容器，则按容器处理：

```text
observe("container")
observe("收纳盒")
observe("粉色盘子")
observe("blue tray")
```

容器 observe 会调用 `PlaceToContainerSkill`：

```text
第一次检测容器 bbox
  -> 估计容器中心
  -> 移动到容器上方 z=450
  -> 第二次检测容器 bbox
  -> 使用 refined center 生成 place plan
  -> 缓存 place_plan
```

透明/半透明收纳盒会被识别为：

```text
container
```

并且不使用颜色筛选。

## grasp(obj)

### 用途

`grasp(obj)` 识别指定药盒并执行抓取。

### 参数

```python
grasp("布洛芬")
grasp("蒙脱石散")
grasp("抗病毒口服液")
grasp("西瓜霜")
```

参数应尽量是药盒 OCR 可读文字或药品名称，不建议使用：

```text
左边药盒
绿色白色药盒
白色蓝色盒子
圆柱形管状物
```

### 当前执行流程

```text
实时拍摄 wrist 图
  -> GroundingDINO 多 prompt 检测药盒候选
  -> OCR 读取候选区域文字
  -> 根据 query 选择目标药盒
  -> 构造 bbox grasp_input
  -> AnyGrasp 在全局点云推理
  -> 用 bbox 筛选 grasp center
  -> 选择满足姿态约束的最高分抓取 best_good
  -> approach
  -> grasp pose
  -> 打开夹爪两次
  -> 关闭夹爪两次
  -> 等待 1.5s
  -> lift
```

### 关键默认参数

```text
grasp_selection_mode = best_good
bbox_filter_inner_margin_ratio = 0.05
collision_detection = True
gripper_open_repeat_count = 2
gripper_close_repeat_count = 2
gripper_close_wait_s = 1.5
```

当前 `grasp` 每次都会重新拍照，不使用任务首帧缓存。

### 调试输出

如果开启：

```bash
--visualize_selected_grasp
--visualize_bbox_filtered_grasps
```

会显示 AnyGrasp 抓取位姿可视化。

执行器还会保存带 bbox 的 wrist 图，例如：

```text
runs/current_capture/bbox_debug_153012_grasp_蒙脱石散_wrist_color.png
```

## place(container)

### 用途

`place(container)` 将当前夹爪中的物体释放到指定容器。

### 参数

```python
place("container")
place("收纳盒")
place("pink tray")
place("blue tray")
```

当前半透明塑料收纳盒推荐使用：

```python
place("container")
```

### 当前执行流程

在任务开始时，`SimpleRobotActions.preobserve_containers()` 会先空手预观测所有需要用到的容器。

```text
预观测容器
  -> 第一次容器检测
  -> 移动到 z=450 二次观察
  -> 第二次容器检测
  -> 缓存 place_plan
```

执行 `place(container)` 时：

```text
读取缓存 place_plan
  -> approach_pose
  -> place_pose
  -> 打开夹爪两次
  -> lift_pose
```

默认不使用 slot 偏移，所有物体放在同一个容器中心点：

```text
enable_place_slot_offsets = False
```

如果以后需要错开放置，可以显式启用：

```bash
--enable_place_slot_offsets
```

### 透明/半透明收纳盒

普通半透明盒子不应带颜色字段。

正确：

```python
place("container")
```

不推荐：

```python
place("pink tray")
place("transparent container")  # transparent 不作为颜色筛选字段
```

透明盒检测使用的 prompt 包括：

```text
empty container
plastic container
storage container
open box container
transparent storage box
translucent storage box
clear plastic container
clear plastic storage bin
open plastic storage box
empty organizer box
open organizer box
empty storage bin
```

## Planner 是如何被提示使用技能的

当前 `PlanningAgent` 的 LLM prompt 已经直接提示模型使用三个执行技能：

```python
observe(...)
grasp(...)
place(...)
```

Prompt 中列出的执行技能是：

```text
observe
grasp
place
```

核心规则包括：

```text
grasp 的 args 必须是 {"target": "药名"}
place 的 args 必须是 {"target": "容器名"}
observe 的 args 必须是 {"target": "药名或容器名"}
常规任务直接生成 grasp -> place
只有失败恢复、目标看不清或需要近距离确认时，才在 grasp 前加入 observe
如果任务要求收纳盒/盒子/半透明盒/透明盒/container/box/bin，且没有明确颜色，
则 place target 必须是 container，不要输出 pink tray
grasp target 必须优先使用药盒上的可读文字或药品名称
禁止把“左边药盒”“绿色白色药盒”等视觉描述当作 medicine_query
```

也就是说，planner 生成的是类似：

```json
{
  "node_id": "n1",
  "node_type": "grasp",
  "skill": "grasp",
  "args": {
    "target": "蒙脱石散"
  }
}
```

```json
{
  "node_id": "n2",
  "node_type": "place",
  "skill": "place",
  "args": {
    "target": "container"
  }
}
```

然后 `SimpleCodeGenerator` 直接渲染成：

```python
grasp("蒙脱石散")
place("container")
```

如果是失败重试 recovery 模式，还会在对应药盒前插入：

```python
observe("蒙脱石散")
grasp("蒙脱石散")
place("container")
```

## 推荐示例

### 单个药盒放入半透明收纳盒

用户输入：

```text
抓住蒙脱石散并放到塑料收纳盒中
```

期望生成：

```python
grasp("蒙脱石散")
place("container")
```

### 多个药盒放入收纳盒

用户输入：

```text
将桌面上的所有药盒收纳进半透明收纳盒中
```

期望生成：

```python
grasp("西瓜霜")
place("container")
grasp("蒙脱石散")
place("container")
grasp("抗病毒口服液")
place("container")
```

### 失败恢复

如果 `place("container")` 失败，且失败药品是 `蒙脱石散`，下一轮恢复执行可能生成：

```python
observe("蒙脱石散")
grasp("蒙脱石散")
place("container")
```

## 相关文件

```text
skill_lib/simple_actions.py              # observe/grasp/place 简短动作定义和执行 facade
agents/simple_code_generator.py          # PlanGraph -> observe/grasp/place
agents/simple_execution_runtime.py       # 真实执行、memory、bbox debug 图保存
agents/planning_agent.py                 # planner prompt 和 PlanGraph 生成
skill_lib/combound/find_and_grasp_medicine.py
skill_lib/combound/place_to_container.py
```
