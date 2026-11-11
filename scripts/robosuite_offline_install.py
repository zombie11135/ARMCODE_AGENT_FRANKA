"""Copy pinned, installed simulation distributions into the dedicated venv.

For the SSD's pre-downloaded packages only. Never imports or modifies the donor
environment, and never copies its editable .pth files or hardware packages.
"""
from importlib.metadata import distributions
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def main():
    root = Path(__file__).resolve().parents[1]
    assert Path(sys.prefix).resolve() == root / '.venv-robosuite', 'Use .venv-robosuite/bin/python'
    site, source = map(Path, sys.argv[1:])
    revision = '5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    normalize = lambda name: name.lower().replace('_', '-')
    available = {normalize(d.metadata['Name']): d for d in distributions(path=[str(site)])}
    pins = ['robosuite==1.5.2'] + [
        row for row in (root / 'requirements-robosuite-backend.txt').read_text().splitlines()
        if '==' in row and not row.startswith('#')
    ]
    selected = []
    for row in pins:
        name, version = row.split('==')
        dist = available[normalize(name)]
        assert dist.version == version, f'{name}: expected {version}, found {dist.version}'
        selected.append(dist)
    dest = Path(sysconfig.get_paths()['purelib'])
    for dist in selected:
        for entry in dist.files:
            # RECORD may also contain scripts outside site-packages. We use -m.
            if '..' in entry.parts or entry.name.startswith('__editable__'):
                continue
            original, copied = site / entry, dest / entry
            if original.is_file():
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
    shutil.copytree(source / 'robosuite', dest / 'robosuite', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print(f'Copied {len(selected)} pinned distributions into {sys.prefix}')


if __name__ == '__main__':
    main()
