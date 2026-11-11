"""GSNet runs with its compatible NumPy/native-extension environment."""
import sys
from unittest.mock import Mock

import pytest

from services import gsnet_service


@pytest.mark.parametrize("configured", [None, "", "/tmp/GSNet env/bin/python"])
def test_worker_interpreter_selection(monkeypatch, tmp_path, configured):
    if configured is None:
        monkeypatch.delenv("GSNET_PICK_PYTHON", raising=False)
    else:
        monkeypatch.setenv("GSNET_PICK_PYTHON", configured)
    run = Mock()
    monkeypatch.setattr(gsnet_service.subprocess, "run", run)
    service = gsnet_service.GSNetService.__new__(gsnet_service.GSNetService)
    request = tmp_path / "request.json"
    with (tmp_path / "worker.log").open("w") as log:
        service._run_inference(request, log)
        command = run.call_args.args[0]
        assert command == [configured or sys.executable,
                           str(gsnet_service.PROJECT_ROOT / "hardware/gsnet_worker.py"),
                           str(request)]
        assert run.call_args.kwargs["check"] is True
        assert run.call_args.kwargs["timeout"] == 180


@pytest.mark.parametrize("configured", [None, "", "/tmp/GSNet env/bin/python"])
def test_saved_grasp_viewer_uses_same_environment(monkeypatch, tmp_path, configured):
    if configured is None:
        monkeypatch.delenv("GSNET_PICK_PYTHON", raising=False)
    else:
        monkeypatch.setenv("GSNET_PICK_PYTHON", configured)
    run = Mock()
    monkeypatch.setattr(gsnet_service.subprocess, "run", run)
    service = gsnet_service.GSNetService.__new__(gsnet_service.GSNetService)
    service.last_run = tmp_path
    service.visualize_last_grasps_by_indices([0])
    assert run.call_args.args[0] == [configured or sys.executable, "-m",
                                    "hardware.gsnet_o3d", str(tmp_path)]
    assert run.call_args.kwargs["cwd"] == gsnet_service.PROJECT_ROOT
