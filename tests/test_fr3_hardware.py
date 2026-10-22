"""Offline tests: no real robot, cameras or serial ports are opened."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hardware.cli import main
from hardware.config import ArmConfig, GripperConfig, HardwareConfig, load_config
from hardware.fr3 import FrankaMoveItArm, MotionError
from hardware.interfaces import PendingTransforms, Pose6D
from hardware.robotiq import GripperError, GripperStatus, Robotiq2F85, crc16
from hardware.runtime import LegacyPoseAdapter, apply_cli_profile, build_hardware, require_task_geometry


def frame(data):
    value = bytes.fromhex(data)
    return value + crc16(value)


def status_frame(flags=0xF9, fault=0, request=255, position=230):
    payload = bytes([9, 3, 6, flags, 0, fault, request, position, 1])
    return payload + crc16(payload)


ACK = bytes.fromhex("09 10 03 E8 00 03 01 30")
TEST_GRIPPERS = []


class SerialDouble:
    def __init__(self, replies):
        self.replies = list(replies)
        self.rx = b""
        self.writes = []
        self.closed = False

    def reset_input_buffer(self):
        self.rx = b""

    def write(self, data):
        self.writes.append(data)
        self.rx = self.replies.pop(0)
        return len(data)

    def read(self, count):
        result, self.rx = self.rx[:count], self.rx[count:]
        return result

    def flush(self):
        pass

    def close(self):
        self.closed = True


def gripper(replies, execute=True):
    serial = SerialDouble(replies)
    factory = Mock(return_value=serial)
    cfg = GripperConfig(port="/dev/test-only", poll_interval_s=0.005)
    driver = Robotiq2F85(cfg, execution_enabled=execute, serial_factory=factory)
    TEST_GRIPPERS.append(driver)
    return driver, serial, factory


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.profile_path = Path(directory.name) / "hardware.json"
        self.profile_path.write_text(json.dumps({"arm": {"workspace": directory.name}}))
        for target in ("hardware.config.DEFAULT_CONFIG", "hardware.cli.DEFAULT_CONFIG"):
            profile_patch = patch(target, self.profile_path)
            profile_patch.start()
            self.addCleanup(profile_patch.stop)
        # Real device configuration may change; offline tests must never open it.
        serial_guard = patch("serial.Serial", side_effect=AssertionError("Real serial access forbidden in offline tests"))
        serial_guard.start()
        self.addCleanup(serial_guard.stop)

    def tearDown(self):
        while TEST_GRIPPERS:
            TEST_GRIPPERS.pop().disconnect()


class ConfigAndPoseTests(OfflineTestCase):
    def test_configured_hardware_construction_is_passive(self):
        load_config()
        with patch("subprocess.Popen", side_effect=AssertionError("must stay passive")):
            hardware = build_hardware()
            hardware.disconnect()

    def test_nonfinite_pose_rejected(self):
        with self.assertRaises(ValueError):
            Pose6D(0, 0, float("nan"), 0, 0, 0)
        with self.assertRaises(ValueError):
            Pose6D(0, 0, 0, 0, 0, 0, target_link="gripper_tcp")

    def test_typos_and_invalid_units_rejected(self):
        for value in ("arm: {speeed_scaling: 0.1}", "arm: {speed_scaling: 2}",
                      "arm: {orientation_tolerance_rad: 90}", "gripper: {slave_id: 0}",
                      "cameras: {global_serial: 123}"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d:
                p = Path(d) / "hardware.yaml"
                p.write_text(value)
                with self.assertRaises(ValueError):
                    load_config(p)

    def test_no_legacy_transform_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "xArm"):
            PendingTransforms().camera_grasp_to_flange_pose(None, "wrist", None)
        with self.assertRaises(RuntimeError):
            require_task_geometry(SimpleNamespace(hardware_config_path=str(self.profile_path)))
        require_task_geometry(SimpleNamespace())  # Explicit old runtime remains available.


class ArmTests(OfflineTestCase):
    def test_pose_command_help_does_not_connect_to_robot(self):
        for operation in ('plan', 'move'):
            with patch('hardware.cli.FrankaMoveItArm') as robot, \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                with self.assertRaises(SystemExit) as stopped:
                    main(['arm', operation, '--help'])
                self.assertEqual(stopped.exception.code, 0)
                self.assertIn('X Y Z RX RY RZ', output.getvalue())
                robot.assert_not_called()

    @staticmethod
    def process_double(ok=True, cleanup=True, interrupt=False):
        def start(command, **kwargs):
            request = json.loads(command[-1])
            result = dict(request_id=request["request_id"], ok=ok, cleanup_ok=cleanup,
                          error="cleanup failed" if not cleanup else "cancelled")
            kwargs["stdout"].write("ARM_AGENT_RESULT=" + json.dumps(result) + "\n")
            kwargs["stdout"].flush()
            child = Mock(returncode=0 if ok else 1)
            child.wait.side_effect = [KeyboardInterrupt(), 0] if interrupt else [0]
            return child
        return start

    def test_cleanup_failure_is_never_success(self):
        with patch("subprocess.Popen", side_effect=self.process_double(cleanup=False)):
            arm = FrankaMoveItArm(ArmConfig())
            with self.assertRaisesRegex(MotionError, "cleanup failed"):
                arm.get_pose()

    def test_gripper_fault_during_motion_cancels_owned_worker(self):
        guard = Mock(side_effect=[None, GripperError('lost communication')])
        arm = FrankaMoveItArm(ArmConfig(), execution_enabled=True, motion_guard=guard)
        with patch('subprocess.Popen', side_effect=self.process_double(ok=False)), \
                patch.object(arm, '_cancel') as cancel:
            with self.assertRaisesRegex(GripperError, 'lost communication'):
                arm.move_pose(Pose6D(.3, 0, .4, 3.14, 0, 0))
        cancel.assert_called_once_with(arm.last_result['request_id'])

    def test_gripper_fault_before_move_never_starts_worker(self):
        arm = FrankaMoveItArm(ArmConfig(), execution_enabled=True,
                             motion_guard=Mock(side_effect=GripperError('lost contact')))
        with patch('subprocess.Popen') as process:
            with self.assertRaisesRegex(GripperError, 'lost contact'):
                arm.move_pose(Pose6D(.3, 0, .4, 3.14, 0, 0))
            process.assert_not_called()

    def test_host_interrupt_signals_owned_remote_request(self):
        arm = FrankaMoveItArm(ArmConfig(), stack="fake", execution_enabled=True)
        with patch("subprocess.Popen", side_effect=self.process_double(ok=False, interrupt=True)), \
                patch.object(arm, "_cancel") as cancel:
            with self.assertRaises(KeyboardInterrupt):
                arm.move_pose(Pose6D(.3, 0, .5, 3.14, 0, 0))
        cancel.assert_called_once_with(arm.last_result["request_id"])

    def test_move_disabled_before_subprocess(self):
        with patch("subprocess.Popen") as popen:
            arm = FrankaMoveItArm(ArmConfig())
            with self.assertRaisesRegex(RuntimeError, "execution_enabled"):
                arm.move_pose(Pose6D(.3, 0, .5, 3.14, 0, -1.57))
            popen.assert_not_called()

    def test_plan_and_move_are_separate_requests(self):
        arm = FrankaMoveItArm(ArmConfig(), execution_enabled=True)
        arm._request = Mock(return_value={"ok": True})
        pose = Pose6D(.3, 0, .5, 3.14, 0, -1.57)
        arm.plan_pose(pose)
        arm._request.assert_called_with("plan", pose)
        arm.move_pose(pose)
        arm._request.assert_called_with("move", pose)

    def test_shell_is_not_used_for_space_containing_path(self):
        arm = FrankaMoveItArm(ArmConfig(workspace="/a path/with (parens)"), stack="fake")
        command = arm._command("print('hello')", "{}")
        self.assertIn("/a path/with (parens)/compose.yaml", command)
        self.assertIn("fake", command)
        self.assertNotIn("real", command)
        self.assertEqual(command[-2:], ["print('hello')", "{}"])

    def test_legacy_units_and_failures(self):
        arm = Mock()
        adapter = LegacyPoseAdapter(arm)
        adapter.set_position(x=350, y=-20, z=500, roll=3.14, pitch=0, yaw=-1.57)
        self.assertEqual(arm.move_pose.call_args.args[0].values, [.35, -.02, .5, 3.14, 0, -1.57])
        arm.get_pose.return_value = Pose6D(.35, -.02, .5, 3.14, 0, -1.57)
        self.assertEqual(adapter.get_position()[1], [350, -20, 500, 3.14, 0, -1.57])
        arm.move_pose.side_effect = MotionError({"error": "aborted"})
        with self.assertRaises(MotionError):
            adapter.set_position(x=350, y=0, z=500, roll=0, pitch=0, yaw=0)

    def test_home_never_uses_xarm_defaults(self):
        with self.assertRaisesRegex(RuntimeError, "home_pose"):
            FrankaMoveItArm(ArmConfig(), execution_enabled=True).home()

    def test_cli_requires_explicit_execution(self):
        for command in (["arm", "move", ".3", "0", ".5", "3.14", "0", "0"],
                        ["gripper", "activate"], ["gripper", "close"]):
            with self.subTest(command=command), patch("sys.stderr", new_callable=io.StringIO), \
                    patch("subprocess.Popen") as popen, self.assertRaises(SystemExit) as error:
                main(command)
            self.assertEqual(error.exception.code, 2)
            popen.assert_not_called()


class RobotiqTests(OfflineTestCase):
    def test_idle_object_bits_do_not_claim_motion_or_contact(self):
        idle = GripperStatus.decode(bytes([0x31, 0, 0, 0, 3, 0]))
        self.assertTrue(idle.ready)
        self.assertEqual(idle.to_dict()["object_state_label"], "not_applicable")
        faulted = GripperStatus.decode(bytes([0xB9, 0, 9, 255, 155, 0]))
        self.assertFalse(faulted.object_detected_closing)
        self.assertEqual(faulted.to_dict()["fault_hex"], "0x09")
        self.assertEqual(faulted.to_dict()["object_state_label"], "not_applicable")

    def test_official_manual_crc_vectors(self):
        self.assertEqual(crc16(bytes.fromhex("09 03 07 D0 00 02")), bytes.fromhex("C5 CE"))
        self.assertEqual(crc16(bytes.fromhex("09 10 03 E8 00 03 06 01 00 00 00 00 00")), bytes.fromhex("72 E1"))
        self.assertEqual(crc16(ACK[:-2]), ACK[-2:])

    def test_status_only_reads_never_activates(self):
        grip, serial, factory = gripper([status_frame()], execute=False)
        factory.assert_not_called()
        state = grip.status()
        self.assertTrue(state.ready)
        self.assertFalse(state.object_detected_closing)
        self.assertEqual([p[1] for p in serial.writes], [3])
        self.assertEqual(serial.writes[0][2:6], bytes.fromhex("07 D0 00 03"))
        self.assertTrue(factory.call_args.kwargs["exclusive"])
        grip.disconnect()
        self.assertTrue(serial.closed)

    def test_no_unconfigured_port_autodiscovery(self):
        factory = Mock()
        with self.assertRaisesRegex(GripperError, "尚未配置"):
            Robotiq2F85(GripperConfig(), serial_factory=factory).status()
        factory.assert_not_called()

    def test_disabled_write_does_not_open_port(self):
        grip, serial, factory = gripper([], execute=False)
        for operation in (grip.activate, grip.open, grip.close, grip.stop):
            with self.assertRaises(GripperError):
                operation()
        factory.assert_not_called()

    def test_unactivated_move_never_autoactivates(self):
        grip, serial, _ = gripper([status_frame(flags=0, request=0, position=0)])
        with self.assertRaisesRegex(GripperError, "activate explicitly"):
            grip.close()
        self.assertEqual([p[1] for p in serial.writes], [3])

    def test_crc_and_exception_fail_closed(self):
        bad = status_frame()[:-2] + b"\x00\x00"
        for reply, message in [(bad, "CRC"), (bytes.fromhex("09 03"), "timeout"),
                               (frame("09 83 02"), "exception code 2")]:
            grip, _, _ = gripper([reply])
            with self.assertRaisesRegex(GripperError, message):
                grip.status()

    def test_write_ack_is_verified_and_stop_attempted(self):
        bad_ack = frame("09 10 03 E9 00 03")
        grip, serial, _ = gripper([status_frame(), bad_ack, ACK])
        with self.assertRaisesRegex(GripperError, "acknowledgement"):
            grip.close()
        self.assertEqual(serial.writes[-1][7], 1)  # rGTO cleared, no release.

    def test_contact_success_requires_new_request_echo(self):
        grip, serial, _ = gripper([
            status_frame(request=0), ACK, status_frame(request=0),
            status_frame(flags=0xB9, request=255, position=155)])
        state = grip.close()
        self.assertTrue(state.object_detected_closing)
        self.assertEqual(state.position_raw, 155)
        self.assertEqual(len(serial.writes), 4)
        self.assertEqual(serial.writes[1][7:13], bytes([9, 0, 0, 255, 64, 64]))

    def test_empty_close_is_not_contact(self):
        grip, _, _ = gripper([status_frame(request=0), ACK, status_frame(position=255)])
        self.assertFalse(grip.close().object_detected_closing)

    def test_fault_stops_without_reset_or_release(self):
        grip, serial, _ = gripper([status_frame(), ACK, status_frame(fault=14), ACK])
        with self.assertRaisesRegex(GripperError, "fault 0x0e"):
            grip.close()
        actions = [p[7] for p in serial.writes if p[1] == 16]
        self.assertEqual(actions, [9, 1])

    def test_activation_is_explicit_and_waits_until_ready(self):
        grip, serial, _ = gripper([status_frame(flags=0), ACK, status_frame(flags=0),
                                   ACK, status_frame(flags=0x11), status_frame(flags=0x31)])
        self.assertTrue(grip.activate().ready)
        self.assertEqual([p[7] for p in serial.writes if p[1] == 16], [0, 1])

    def test_major_fault_is_not_reset_by_activate(self):
        grip, serial, _ = gripper([status_frame(fault=13)])
        with self.assertRaises(GripperError):
            grip.activate()
        self.assertEqual([p[1] for p in serial.writes], [3])


class AgentWiringTests(OfflineTestCase):
    def test_config_replaces_old_machine_paths(self):
        from hardware.config import DEFAULT_CONFIG
        args = SimpleNamespace(legacy_xarm=False, hardware_config=str(DEFAULT_CONFIG),
                               simple_actions=False, xarm_ip="192.168.1.237", handeye_config="old.yaml")
        apply_cli_profile(args)
        self.assertTrue(args.simple_actions)
        self.assertEqual(args.xarm_ip, "")
        self.assertEqual(args.handeye_config, load_config(DEFAULT_CONFIG).calibration_file or "")
        self.assertEqual(args.global_serial, "346522072096")
        self.assertFalse(args.ignore_grip_error)

    def test_legacy_mode_is_explicit(self):
        args = SimpleNamespace(legacy_xarm=True, hardware_config="bad file", simple_actions=False)
        apply_cli_profile(args)
        self.assertIsNone(args.hardware_config)
        self.assertFalse(args.simple_actions)

    def test_runtime_stops_before_models_cameras_or_motion(self):
        from agents.simple_agent_system import SimpleAgentSystemConfig, SimpleArmCodeAgentSystem
        from agents.simple_execution_runtime import SimpleExecutionRuntime
        from hardware.config import DEFAULT_CONFIG
        cfg = SimpleAgentSystemConfig(goal="test", hardware_config_path=str(DEFAULT_CONFIG), execute_robot=True)
        runtime = SimpleExecutionRuntime(cfg)
        with patch("hardware.runtime.load_transforms", return_value=PendingTransforms()), patch.object(runtime, "_build_execution_context") as make_context:
            with self.assertRaises(RuntimeError):
                runtime.run([])
            make_context.assert_not_called()
        agent = SimpleArmCodeAgentSystem(cfg)
        with patch("hardware.runtime.load_transforms", return_value=PendingTransforms()), patch.object(agent, "_planning_stage") as planner:
            with self.assertRaises(RuntimeError):
                agent.run()
            planner.assert_not_called()

    def test_builtin_fr3_never_calls_legacy_xarm_or_ros1(self):
        from hardware.config import DEFAULT_CONFIG
        from skill_lab.run_agent_system import run_builtin_command
        args = SimpleNamespace(hardware_config=str(DEFAULT_CONFIG), execute_robot=True)
        with patch("skill_lab.run_agent_system.init_arm", side_effect=AssertionError("legacy arm")), \
                patch("skill_lab.run_agent_system.call_grip_service", side_effect=AssertionError("ROS 1")), \
                patch("hardware.runtime.load_config", return_value=HardwareConfig()), \
                patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertTrue(run_builtin_command("/open", args))
            self.assertIn("尚未配置", out.getvalue())
            self.assertTrue(run_builtin_command("/home", args))
            self.assertIn("home_pose", out.getvalue())


if __name__ == "__main__":
    unittest.main()
