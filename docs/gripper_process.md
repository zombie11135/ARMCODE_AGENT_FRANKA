# FR3 Agent 夹爪通信进程

2026-10-22：正式 FR3 Agent 使用 `ProcessGripper`，将串口轮询和原有开合、接触闭合策略放入独立 Python 进程。主进程继续运行感知、技能调度和机械臂控制。构造对象只校验配置；建立会话才启动子进程、连接串口。

此前轮询线程与感知共用 Python 解释器锁（GIL）。某段本地计算长时间占用锁时，轮询线程可能无法及时运行。原驱动会在相邻状态更新超过 `state_timeout_s=0.5` 秒时锁定失败，即使随后又收到正常状态也不会解除。单独运行 open/close 时没有同样的感知负载，因而可能正常。

旧失败报告没有记录首次间隔的具体耗时，无法据此证明现场一定是 GIL 阻塞。离线测试已经复现这类问题，并验证独立进程能在主进程持锁 0.75 秒期间继续轮询。USB、串口和系统整体调度故障仍需通过新诊断区分，现场尚待复测。

## 使用

退出旧 Agent 后，使用此前完整的 `run_fr3_agent.sh` 启动命令，保留原任务、清障、识别和缓冲位参数；无需新增开关、安装模型或修改硬件参数。启动执行会话后应看到：

```text
[FR3] 夹爪通信: process worker_pid=<PID> state_timeout_s=0.5
```

原 Agent 的开爪、闭合、抬升回调和任务结束后的 status/open/stop/quit 终端均经此接口调用。`hardware.cli` 和独立接触闭合实验继续使用原驱动。

子进程内按原配置选用 `AgentContactGripper` 或 `Robotiq2F85`，不修改闭合算法。本机原接触配置仍为 100 mA 电流上限、接触后最多追加 2 raw。达到电流上限立即停止；停止确认且设备健康时计为闭合完成，`grasp_verified` 仍为 false。

## 故障与退出

- 保留 0.5 秒状态间隔检查和原有接触策略采样间隔检查；真实超时仍拒绝后续动作。
- 串口错误、进程退出、RPC 超时均不自动重连、不重发动作。RPC 超时意味着执行结果可能未知。
- stop 使用独立调用连接，可中断另一个线程正在等待的闭合命令；不会排在该闭合调用之后。
- 正常退出关闭串口和通信进程，不自动张开。Agent 异常退出时，子进程检测父进程消失，若有在途命令则尽力停止，再关闭串口；通信失效时不能保证停止送达。
- 出错后需操作者检查并显式断开、重新启动会话。新进程不会接管另一个程序已经占用的串口。

## 诊断

`gripper.session_info()` 包含：

| 字段 | 含义 |
| --- | --- |
| `communication_mode`, `worker_pid`, `worker_alive` | 通信方式及进程状态 |
| `last_status_gap_s`, `max_status_gap_s` | 相邻成功状态更新的间隔 |
| `last_status_transaction_s` | 最近一次状态请求的串口收发耗时 |
| `last_status_lock_wait_s` | 最近一次状态请求等待串口锁的耗时 |
| `first_fault` | 首次失败的时间、间隔、耗时、阈值和原始状态；后续正常读数不会覆盖 |
| `transport_fault` | RPC 失败时的方法、等待时长、期限和错误类型 |

清障异常会将上述信息写入 `runs/current_plan/failure_report.json` 的 `evidence.clear_result.gripper_session`。可据此判断是状态更新间隔过大、串口事务缓慢，还是通信进程/RPC 失败。不要只根据随后返回的 ready 状态认定会话已恢复。

## 离线验证

使用项目测试环境执行，测试全部使用模拟串口、合成图像与机械臂替身，不访问真实设备：

```bash
.venv-mobileclip2-fr3/bin/python -m pytest -q tests/test_gripper_process.py
.venv-mobileclip2-fr3/bin/python scripts/check_core.py
```

测试覆盖主进程 GIL 阻塞、原 Agent 回调到闭合后抬升、100 mA 停止完成语义、串口故障锁定、RPC 超时不重试、通信进程丢失、并发 stop 和父进程异常退出清理。离线通过不等于真机任务已经完成。
