import unittest
from unittest.mock import patch
import struct

from hardware.config import GripperConfig
from hardware.robotiq import GripperError, GripperStatus
from hardware.robotiq_diagnostics import StatusOnlyGripper, check_communication, summarize


def sample(t, fault=0, ready=True):
    flags = 0x31 if ready else 0
    return dict(elapsed_s=t, latency_ms=10,
                status=GripperStatus.decode(bytes([flags, 0, fault, 0, 3, 0])).to_dict())


class DiagnosticsTests(unittest.TestCase):
    def test_transport_forbids_write_and_other_addresses(self):
        reader = StatusOnlyGripper(GripperConfig(port="/dev/never-open"))
        with patch("serial.Serial", side_effect=AssertionError("must not open")) as serial:
            for function, payload in [(16, b""), (3, struct.pack(">HH", 0x07D0, 2)),
                                      (3, struct.pack(">HH", 0x03E8, 3))]:
                with self.assertRaises(GripperError):
                    reader._exchange(payload, function)
            with self.assertRaises(GripperError):
                reader.activate()
            serial.assert_not_called()

    def test_startup_timeout_is_distinguished_from_steady_fault(self):
        rows = [sample(0, 9)] + [sample(i / 10) for i in range(1, 31)]
        report = summarize(rows, warmup_s=2, duration_s=3, interval_s=.1, completed=True)
        self.assertTrue(report["communication_passed"])
        self.assertEqual(report["fault_counts"], {"0x09": 1, "0x00": 30})
        self.assertEqual(report["first_fault_free_s"], .1)
        rows[-1] = sample(3, 9)
        report = summarize(rows, warmup_s=2, duration_s=3, interval_s=.1, completed=True)
        self.assertFalse(report["communication_passed"])
        self.assertEqual(report["steady_nonzero_fault_samples"], 1)

    def test_crc_error_and_interruption_cannot_pass(self):
        rows = [sample(i / 10) for i in range(31)]
        for changed, completed in [(rows + [dict(elapsed_s=3.1, error="CRC mismatch")], True), (rows, False)]:
            result = summarize(changed, warmup_s=2, duration_s=3, interval_s=.1, completed=completed)
            self.assertFalse(result["communication_passed"])

    def test_inactive_gripper_is_not_communication_failure(self):
        result = summarize([sample(i / 10, ready=False) for i in range(31)],
                           warmup_s=2, duration_s=3, interval_s=.1, completed=True)
        self.assertTrue(result["communication_passed"])
        self.assertFalse(result["all_steady_samples_ready"])

    def test_invalid_timing_fails_before_connecting(self):
        with patch("serial.Serial", side_effect=AssertionError("must not open")) as serial:
            for duration, interval in [(0, .1), (float("inf"), .1), (3, .001), (3, float("nan"))]:
                with self.assertRaises(ValueError):
                    check_communication(GripperConfig(), duration_s=duration, interval_s=interval)
            serial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
