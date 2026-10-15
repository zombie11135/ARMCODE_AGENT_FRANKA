"""Interactive gripper session. Hardware actions are entered by the operator."""
from datetime import datetime
import json
from pathlib import Path
import shlex
from zoneinfo import ZoneInfo

from .config import PROJECT_ROOT
from .robotiq import Robotiq2F85


HELP = """Commands:
  status                 读取状态和后台通信状态
  activate               显式激活（未激活时会执行自标定运动）
  open [speed force]     打开；可选速度/力度均为 0..255 协议值
  close [speed force]    闭合；空夹到位不等于抓到物体
  move raw [speed force] 指定 0..255 位置，0 打开、255 闭合
  stop                   停止夹爪运动
  reconnect              关闭/重开串口并只读建立通信，不复位、不重试动作
  help / quit            帮助 / 退出（退出不自动打开夹爪）
等待输入期间保持 10 Hz 只读轮询（频率以配置为准）。不要同时运行其它串口客户端。
"""


def raw_byte(value):
    number = int(value)
    if not 0 <= number <= 255:
        raise ValueError("Position, speed and force must be integer protocol values in [0, 255]")
    return number


def dispatch(gripper, words):
    command, *args = words
    if command in ("status", "activate", "stop", "reconnect"):
        if args:
            raise ValueError(f"{command} takes no arguments")
        if command == "reconnect":
            gripper.disconnect()
            return gripper.start_session()
        return getattr(gripper, command)()
    if command in ("open", "close", "move"):
        position = None
        if command == "move":
            if not args:
                raise ValueError("Usage: move POSITION [SPEED FORCE]")
            position, args = raw_byte(args[0]), args[1:]
        if len(args) not in (0, 2):
            raise ValueError(f"Usage: {command} " + ("POSITION " if command == "move" else "") + "[SPEED FORCE]")
        options = dict(zip(("speed_raw", "force_raw"), map(raw_byte, args)))
        return gripper.move_raw(position, **options) if command == "move" else getattr(gripper, command)(**options)
    raise ValueError(f"Unknown command: {command}; use help")


def run_console(config, *, execution_enabled=False, output_dir=None, input_fn=input, output_fn=print):
    folder = Path(output_dir) if output_dir else PROJECT_ROOT / "runs/gripper"
    folder.mkdir(parents=True, exist_ok=True)
    name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("session-%Y%m%d-%H%M%S-%f.jsonl")
    path = folder / name
    output_fn(f"Log: {path}")
    output_fn("动作已启用：仅执行你输入的动作。" if execution_enabled else "只读会话：动作命令将被拒绝；--execute 才启用动作。")
    with path.open("x") as log:
        def record(event, **data):
            log.write(json.dumps(dict(time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                      event=event, **data), ensure_ascii=False) + "\n")
            log.flush()

        gripper = Robotiq2F85(config, execution_enabled=execution_enabled)
        try:
            state = gripper.start_session()
            record("session_started", status=state.to_dict(), session=gripper.session_info())
            output_fn(json.dumps(dict(status=state.to_dict(), session=gripper.session_info()), ensure_ascii=False, indent=2))
            output_fn(HELP)
            while True:
                try:
                    line = input_fn("robotiq> ")
                except (EOFError, KeyboardInterrupt):
                    break
                try:
                    words = shlex.split(line)
                    if not words:
                        continue
                    if words[0] in ("quit", "exit") and len(words) == 1:
                        break
                    if words == ["help"]:
                        output_fn(HELP)
                        continue
                    record("command", text=line)
                    state = dispatch(gripper, words)
                    data = dict(ok=True, command=words[0], status=state.to_dict(), session=gripper.session_info())
                    record("result", **data)
                    output_fn(json.dumps(data, ensure_ascii=False, indent=2))
                except (Exception, KeyboardInterrupt) as error:
                    data = dict(ok=False, error=repr(error), session=gripper.session_info())
                    record("command_failed", **data)
                    output_fn(json.dumps(data, ensure_ascii=False, indent=2))
        except BaseException as error:
            record("session_failed", error=repr(error))
            raise
        finally:
            gripper.disconnect()
            record("session_closed", session=gripper.session_info())
    output_fn("会话已关闭；后台轮询已停止。")
    return {"ok": True, "log_path": str(path), "execution_enabled": execution_enabled}
