"""The snapshot must include new source and exclude local models/history."""
from pathlib import Path

import pytest

from scripts.prepare_release import audit, export, source_files


def write(root, name, content="example\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_export_keeps_new_source_and_templates_only(tmp_path):
    root = tmp_path / "workspace"
    write(root, "app/new_untracked.py", "print('hello')\n")
    write(root, "config/hardware/fr3.example.yaml")
    for name in (".git/config", "config/hardware/fr3.yaml", "config/local.env",
                 "data/gallery.json", "dependencies/sdk/setup.py", "weights/model.pt",
                 "services/runtime.so", "services/licenseCfg.json",
                 "anyplace_real/config/real_saved_xarm_snapshot.json"):
        write(root, name)
    destination = tmp_path / "release"
    # licenseCfg.json is rejected as an unapproved credential file below.
    assert export(root, destination) == 2
    assert sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()) == [
        "RELEASE_MANIFEST.json", "app/new_untracked.py", "config/hardware/fr3.example.yaml",
    ]
    with pytest.raises(ValueError, match="already exist"):
        export(root, destination)


def test_credential_blocks_export_without_printing_value(tmp_path):
    token = "sk-" + "x" * 30
    root = tmp_path / "workspace"
    write(root, "app/cli.py", "key = '" + token + "'\n")
    destination = tmp_path / "release"
    with pytest.raises(ValueError, match="possible credential") as error:
        export(root, destination)
    assert token not in str(error.value)
    assert not destination.exists()


def test_symlinks_and_tracked_local_files_rejected(tmp_path):
    target = write(tmp_path, "private.txt")
    link = tmp_path / "app" / "leak.py"
    link.parent.mkdir()
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        audit(tmp_path, source_files(tmp_path))
    with pytest.raises(ValueError, match="boundary"):
        audit(tmp_path, [Path("config/local.env")])


def test_export_includes_gallery_assets_without_caches_or_other_data(tmp_path):
    import hashlib
    import json

    root = tmp_path / "workspace"
    assets = ("data/object_gallery/gallery.json",
              "data/object_gallery/images/reference.png",
              "data/object_gallery/images/reference.jpg",
              "data/object_gallery/captures/sample/color.png",
              "data/object_gallery/captures/sample/capture.json")
    for name in assets:
        path = write(root, name, "{}" if name.endswith(".json") else "")
        if not name.endswith(".json"):
            path.write_bytes(b"\x89PNG\r\n\x1a\n\xff")
    excluded = ("data/other/private.json", "data/object_gallery/.lock",
                "data/object_gallery/features/cache.npz",
                "data/object_gallery/images/private.json",
                "data/object_gallery/captures/sample/depth.npz",
                "data/object_gallery/captures/sample/.env",
                "services/data/private.json")
    for name in excluded:
        write(root, name)
    destination = tmp_path / "release"
    assert export(root, destination) == len(assets)
    manifest = json.loads((destination / "RELEASE_MANIFEST.json").read_text())["sha256"]
    assert set(manifest) == set(assets)
    for name in assets:
        assert (destination / name).read_bytes() == (root / name).read_bytes()
        assert manifest[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in excluded:
        assert not (destination / name).exists()
        with pytest.raises(ValueError, match="boundary"):
            audit(root, [Path(name)])
