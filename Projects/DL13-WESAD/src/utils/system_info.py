"""Capture the actual runtime and dependency versions for each group."""
import importlib.metadata
import platform
import subprocess
import sys
import psutil
import torch


def system_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in ['torch', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'matplotlib', 'psutil', 'pytest', 'tensorboard']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(), 'cpu': platform.processor(),
            'cpu_count': psutil.cpu_count(), 'ram_total': psutil.virtual_memory().total,
            'ram_available': psutil.virtual_memory().available, 'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'packages': packages, 'git_commit': commit, 'git_dirty': dirty}
