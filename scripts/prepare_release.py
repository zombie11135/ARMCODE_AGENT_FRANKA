"""Export a source snapshot without Git history, vendor code or machine-local assets.

Only Python's standard library is required. The existing checkout is never
changed. A path allowlist defines the release boundary; extend it deliberately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {
    "agents", "app", "core", "runtime", "services", "skill_lib", "skill_lab",
    "hardware", "perception_v2", "nesyro_gate", "gate_learning", "gate_sim",
    "anyplace_real", "robocasa_occlusion", "occlusion_reasoning",
    "tests", "scripts", "docs", ".github",
}
ROOT_FILES = {
    ".gitignore", "README.md", "CONTRIBUTING.md", "THIRD_PARTY.md", "LICENSE",
    "NOTICE", "main_agent.py", "test_xarm.py", "RELEASE_MANIFEST.json",
    "run_fr3_agent.sh", "run_anygrasp_offline.sh", "run_gsnet_pick.sh",
    "view_gsnet_o3d.sh", "register_object_photos.sh",
    "capture_wrist_photos.sh",
    "run_robocasa_scene.sh", "run_occlusion_camera_test.sh",
    "calibrate_overhead_camera.sh", "test_aruco_pose.sh",
    "test_gripper_contact.sh",
    "ANYPLACE_REAL_STAGE1_REPORT.md", "ANYPLACE_REAL_STAGE2_REPORT.md",
    "PERCEPTION_V2_INITIAL_REPORT.md",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".dot", ".svg"}
EXTRA_FILES = {
    "config/local.env.example", "config/hardware/fake.override.yaml",
    "docs/multimodal_gate_architecture.png",
    "skill_lab/test_segment_obj",
}
EXCLUDED_DIRS = {
    ".git", ".claude", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    "dependencies", "third_party", "external", "models", "weights", "checkpoints",
    "anygrasp_sdk", "runs", "logs", "outputs", "data", "license", "build", "dist",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^\s/@:'\"]+:[^\s/@'\"]+@"),
)


def allowed(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    gallery = relative.parts[:2] == ("data", "object_gallery")
    checked_parts = relative.parts[1:] if gallery else relative.parts
    if any(p in EXCLUDED_DIRS or p.startswith(".venv") for p in checked_parts):
        return False
    name = relative.name
    if name == "licenseCfg.json" or name.endswith((".lic", ".key", ".pem", ".public_key", ".signature")):
        return False
    if name.startswith(".env") or ".local." in name or ".private." in name:
        return False
    if gallery:
        if relative.as_posix() == "data/object_gallery/gallery.json":
            return True
        if len(relative.parts) == 4 and relative.parts[2] in {"images", "raw_images"}:
            return relative.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        return (len(relative.parts) == 5 and relative.parts[2] == "captures"
                and name in {"color.png", "capture.json"})
    if relative.as_posix() in EXTRA_FILES:
        return True
    if relative.parts[0] == "config":
        return name.endswith((".example.yaml", ".example.json"))
    if relative.parts[:2] == ("anyplace_real", "config"):
        return name.endswith(".example.json")
    if len(relative.parts) == 1:
        return name in ROOT_FILES or (name.startswith("requirements-") and name.endswith(".txt"))
    return relative.parts[0] in SOURCE_DIRS and relative.suffix in TEXT_SUFFIXES


def source_files(root):
    selected = []
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = sorted(n for n in children
                             if (n not in EXCLUDED_DIRS or (Path(directory) == root and n == "data"))
                             and not n.startswith(".venv"))
        # Do not traverse directories outside the explicitly maintained scope.
        if Path(directory) == root:
            children[:] = [n for n in children if n in SOURCE_DIRS or n in {"config", "data"}]
        elif Path(directory) == root / "data":
            children[:] = [n for n in children if n == "object_gallery"]
        for name in sorted(names):
            relative = (Path(directory) / name).relative_to(root)
            if allowed(relative) and name != "RELEASE_MANIFEST.json":
                selected.append(relative)
    return sorted(selected)


def audit(root, files):
    errors = []
    for relative in files:
        path = root / relative
        if not allowed(relative):
            errors.append(f"{relative}: outside the source release boundary")
            continue
        if any(p.is_symlink() for p in (path, *path.parents)):
            errors.append(f"{relative}: symlinks are not exported")
            continue
        if not path.is_file():
            errors.append(f"{relative}: missing regular file")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"{relative}: exceeds 5 MiB source-file limit")
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            match = pattern.search(data)
            if match:
                line = data[:match.start()].count(b"\n") + 1
                # Never echo matched text or token values.
                errors.append(f"{relative}:{line}: possible credential (redacted)")
                break
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"{relative}: unexpected binary content")
    if errors:
        raise ValueError("\n".join(errors))


def export(root, destination):
    root, destination = root.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output must not already exist; choose a new directory.")
    if destination.resolve().is_relative_to(root):
        raise ValueError("Output must be outside the current source checkout.")
    files = source_files(root)
    audit(root, files)
    if not files:
        raise ValueError("No source files selected.")
    destination.mkdir(parents=True)
    manifest = {}
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "RELEASE_MANIFEST.json").write_text(
        json.dumps({"format": 1, "sha256": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit(destination, files)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="new directory outside this checkout")
    mode.add_argument("--check-tracked", action="store_true", help="audit Git's current index (CI / after git add)")
    mode.add_argument("--check", action="store_true", help="audit the files selected for export")
    args = parser.parse_args()
    try:
        if args.output:
            count = export(ROOT, args.output.expanduser())
            print(f"Exported {count} source files to {args.output}; no Git history included.")
        else:
            if args.check_tracked:
                if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
                    raise ValueError("Tracked files differ from the index; review and stage changes before checking.")
                result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
                files = [Path(os.fsdecode(p)) for p in result.stdout.split(b"\0") if p]
                if not files:
                    raise ValueError("Git index is empty; stage the source files first.")
            else:
                files = source_files(ROOT)
            audit(ROOT, files)
            print(f"Source boundary and credential-pattern checks passed: {len(files)} files.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
