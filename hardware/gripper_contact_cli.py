"""Standalone experimental console; real motion only after an operator command."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
import sys

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .gripper_contact import ContactExperimentGripper, ContactSettings, EmptyBaseline, export_trial


HELP = """
status                    读取夹爪状态
activate                  仅未激活时执行激活自标定（会运动）
open                      打开夹爪（会释放已夹持物体）
baseline                  空载闭合采集；先清空夹爪并 open，不自动打开
grasp                     基线接触实验；先 open，再放入物体
set contact-rise-ma 30     高于空载基线的电流增量，mA
set extra-close-raw 1      接触后最多追加的位置协议格数；0=接触即停
set current-limit-ma 100   闭合电流上限，mA；不是牛顿
set confirm-samples 2      连续超基线的采样数
set speed-raw 0            最低速度；改变后重新采集 baseline
set force-raw 0            最低力度；改变后重新采集 baseline
set max-close-raw 225      总闭合行程终点；改变后重新采集 baseline
settings                  查看当前实验参数
stop                      发送停止；正在实验时用 Ctrl+C 中断
plot                      从上次实验日志导出 CSV / PNG，不打开窗口
help / quit               帮助 / 退出；退出不自动打开夹爪

只控制夹爪，不连接机械臂或相机。启动不会自动激活或运动。
先 baseline，随后每次 open -> 手动放入物体 -> grasp。
raw 越大越闭合；1 格约 0.4 mm，仅为名义分辨率，未做本机毫米标定。
追加位移包含接触确认期间已走过的距离；停车仍可能超调，查看 overshoot_raw。
电流接触只是估计，不判定抓取成功，不自动抬升或再次夹紧。
"""


def summary(report):
    keys = ('mode', 'outcome', 'contact', 'target_raw', 'closure_after_contact_raw',
            'overshoot_raw', 'peak_current_ma', 'max_sample_gap_ms', 'sample_count',
            'stop_latency_ms', 'contact_to_stop_ms',
            'stop_confirmed', 'grasp_verified', 'error', 'stop_error', 'log_path')
    return {k: report[k] for k in keys if k in report}


def console(config, settings, *, execute=False, output_dir, baseline=None,
            input_fn=input, output_fn=print, driver_factory=ContactExperimentGripper):
    settings.validate()
    folder = Path(output_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    driver = driver_factory(config, execution_enabled=execute)
    output_fn('动作已启用；仅执行你输入的夹爪命令。' if execute else '只读会话；动作需要 --execute。')
    output_fn('当前数值是实验参数，电流没有转换成牛顿。不要同时运行 Agent 或其它夹爪串口客户端。')
    try:
        output_fn(json.dumps(driver.start_session().to_dict(), ensure_ascii=False))
        output_fn(HELP)
        output_fn(json.dumps(asdict(settings), indent=2))
        while True:
            try:
                words = shlex.split(input_fn('contact> '))
            except (EOFError, KeyboardInterrupt):
                break
            except ValueError as error:
                output_fn(str(error))
                continue
            if not words:
                continue
            before = driver.last_trial
            try:
                command, *args = words
                if command in ('quit', 'exit') and not args:
                    break
                if command == 'set':
                    if len(args) != 2:
                        raise ValueError('Usage: set PARAMETER VALUE')
                    name = args[0].replace('-', '_')
                    if name not in ('contact_rise_ma', 'extra_close_raw', 'current_limit_ma',
                                    'confirm_samples', 'speed_raw', 'force_raw', 'max_close_raw'):
                        raise ValueError('Unknown setting; use help')
                    proposed = replace(settings, **{name: int(args[1])})
                    proposed.validate()
                    if config.close_current_limit_ma is not None and proposed.current_limit_ma > config.close_current_limit_ma:
                        raise ValueError('Cannot exceed hardware profile current ceiling')
                    if name in ('speed_raw', 'force_raw', 'max_close_raw') and proposed != settings:
                        baseline = None
                        output_fn('运动条件改变，基线已清除；需要重新采集 baseline。')
                    settings = proposed
                    output_fn(json.dumps(asdict(settings)))
                elif args:
                    raise ValueError(f'{command} takes no arguments')
                elif command == 'help':
                    output_fn(HELP)
                elif command == 'settings':
                    output_fn(json.dumps(dict(settings=asdict(settings), baseline_source=None if baseline is None else baseline.source_log), indent=2))
                elif command in ('status', 'activate', 'stop', 'open'):
                    options = dict(speed_raw=settings.speed_raw, force_raw=settings.force_raw) if command == 'open' else {}
                    output_fn(json.dumps(getattr(driver, command)(**options).to_dict(), ensure_ascii=False))
                elif command in ('baseline', 'grasp'):
                    if command == 'baseline':
                        baseline = None  # Never retain a baseline after a failed recalibration.
                    output_fn('正在执行空载闭合采集…' if command == 'baseline' else '正在执行接触后限位闭合…')
                    report = driver.run_trial(settings, baseline=baseline, mode=command, output_dir=folder)
                    output_fn(json.dumps(summary(report), ensure_ascii=False, indent=2))
                    if command == 'baseline':
                        baseline = EmptyBaseline.from_trial(report)
                        baseline_path = folder / (Path(report['log_path']).stem+'.baseline.json')
                        baseline_path.write_text(json.dumps(asdict(baseline), indent=2))
                        output_fn(f'基线已保存：{baseline_path}\n下一步 open，再放入物体，然后 grasp。')
                    else:
                        output_fn('实验已停止；不代表已夹稳。open 会释放物体，quit 不会自动打开。')
                elif command == 'plot':
                    if driver.last_trial is None:
                        raise ValueError('No trial yet')
                    output_fn(json.dumps(export_trial(driver.last_trial), ensure_ascii=False))
                else:
                    raise ValueError('Unknown command; use help')
            except (Exception, KeyboardInterrupt) as error:
                output_fn(f'Stopped: {error!r}; no automatic retry/open.')
                if driver.last_trial is not None and driver.last_trial is not before:
                    output_fn(json.dumps(summary(driver.last_trial), ensure_ascii=False, indent=2))
            finally:
                report = driver.last_trial
                if report is not None and report is not before:
                    # Never spend time plotting while a stop is unconfirmed.
                    if report.get('stop_confirmed'):
                        try:
                            output_fn(json.dumps(export_trial(report), ensure_ascii=False))
                        except Exception as error:
                            output_fn(f'Plot export failed (JSON retained): {error}')
                    if not report.get('stop_confirmed'):
                        output_fn('未确认停止；结束本次会话，请检查夹爪状态后重新连接。')
                        break
    finally:
        driver.disconnect()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Robotiq-only contact + bounded extra closure experiment')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--port', help='Override explicit RS-485 /dev/... port')
    parser.add_argument('--execute', action='store_true', help='Enable actions typed at the console; no automatic motion')
    parser.add_argument('--current-limit-ma', type=int, help='Required for console: explicit experimental current ceiling')
    parser.add_argument('--contact-rise-ma', type=int, default=30)
    parser.add_argument('--extra-close-raw', type=int, default=1)
    parser.add_argument('--confirm-samples', type=int, default=2)
    parser.add_argument('--max-close-raw', type=int, default=225)
    parser.add_argument('--speed-raw', type=int, default=0)
    parser.add_argument('--force-raw', type=int, default=0)
    parser.add_argument('--interval', type=float, default=.01, help='Requested sleep between polls, actual timing is logged')
    parser.add_argument('--baseline', help='Reuse a matching *.baseline.json (same gripper/setup)')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT/'runs/gripper_contact'))
    parser.add_argument('--plot', metavar='REPORT_JSON', help='Offline plot only; does not load hardware config or open ports')
    args = parser.parse_args(argv)
    if args.plot:
        report = json.loads(Path(args.plot).expanduser().read_text())
        report['log_path'] = str(Path(args.plot).expanduser().resolve())
        print(json.dumps(export_trial(report), ensure_ascii=False, indent=2))
        return 0
    if args.current_limit_ma is None:
        parser.error('--current-limit-ma is required; it is an experimental ceiling, not calibrated force')
    if not .005 <= args.interval <= .1:
        parser.error('--interval must be 0.005..0.1 seconds')
    try:
        settings = ContactSettings(**{k: getattr(args, k) for k in ('current_limit_ma', 'contact_rise_ma',
            'extra_close_raw', 'confirm_samples', 'max_close_raw', 'speed_raw', 'force_raw')})
        settings.validate()
        cfg = load_config(args.config).gripper
        if cfg.close_current_limit_ma is not None and settings.current_limit_ma > cfg.close_current_limit_ma:
            raise ValueError('Experiment ceiling exceeds hardware profile current ceiling')
        cfg = replace(cfg, port=args.port or cfg.port, poll_interval_s=args.interval,
                      keepalive_interval_s=min(.05, cfg.keepalive_interval_s))
        if not cfg.port or not cfg.port.startswith('/dev/'):
            raise ValueError('Configure an explicit gripper /dev/... port first')
        baseline = None
        if args.baseline:
            baseline = EmptyBaseline(**json.loads(Path(args.baseline).expanduser().read_text()))
            baseline.check(cfg, settings, baseline.start_raw)
        return console(cfg, settings, execute=args.execute, output_dir=args.output_dir, baseline=baseline)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'Stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
