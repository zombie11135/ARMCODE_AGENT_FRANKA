"""Download the official asset subsets needed by the fixed kitchen layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("dependencies/robocasa365"))
    parser.add_argument("--cache", type=Path, default=Path("dependencies/robocasa-downloads"))
    args = parser.parse_args()
    root = (args.repo / "robocasa/models/assets").resolve()
    links = json.loads((root / "box_links/box_links_assets.json").read_text())
    expected = json.loads(Path(__file__).with_name("upstream.lock.json").read_text())["asset_archive_sha256"]
    args.cache.mkdir(parents=True, exist_ok=True)
    for key, name, destination in [
        ("textures", "textures", root),
        ("generative_textures", "generative_textures", root),
        ("fixtures_lightwheel", "fixtures", root),
        ("objects_lightwheel", "lightwheel", root / "objects"),
    ]:
        url = links[key].replace("/s/", "/shared/static/") + ".zip"
        archive = args.cache / f"{name}.zip"
        marker = args.cache / f"{name}.extracted.json"
        if marker.exists() and (destination / name).is_dir():
            recorded = json.loads(marker.read_text())
            if recorded.get("sha256") == expected[name] and recorded.get("destination") == str(destination):
                print(f"Already extracted: {name}", flush=True)
                continue
        if not archive.exists():
            partial = archive.with_suffix(".zip.partial")
            print(f"Downloading official {name} assets: {url}", flush=True)
            # Use GET: some official Box links return 404 to HEAD but serve GET.
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
            partial.replace(archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            raise ValueError(f"Official {name} archive differs from the verified version: {digest}")
        with ZipFile(archive) as zipped:
            for member in zipped.namelist():
                if not (destination / member).resolve().is_relative_to(destination):
                    raise ValueError(f"Archive path escapes destination: {member}")
            print(f"Extracting {name}: {len(zipped.namelist())} entries", flush=True)
            zipped.extractall(destination)
        marker.write_text(json.dumps({"url": url, "sha256": digest, "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
