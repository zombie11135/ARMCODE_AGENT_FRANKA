"""Local VLM service lifecycle. No camera, robot or inference imports on import."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'qwen35-9b-awq': dict(repo='cyankiwi/Qwen3.5-9B-AWQ-4bit',
        revision='e4d2d39cede8c5d1716d824cd843e2889c648083', gpu_fraction=.45),
    'qwen3vl-8b-awq': dict(repo='cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit',
        revision='914bc902ca94c0135be5aa70f25a616bfc853140', gpu_fraction=.38),
}


def model_root():
    return Path(os.environ.get('ARM_LOCAL_VLM_ROOT', ROOT/'models/local-vlm')).expanduser().resolve()


def service_environment():
    env = os.environ.copy()
    # Select the server GPU without changing the Agent's perception device.
    if env.get('ARM_LOCAL_VLM_CUDA_VISIBLE_DEVICES'):
        env['CUDA_VISIBLE_DEVICES'] = env['ARM_LOCAL_VLM_CUDA_VISIBLE_DEVICES']
    cache = Path(env.get('ARM_LOCAL_VLM_CACHE', ROOT/'runs/local_vlm/cache')).resolve()
    for key, name in [('HF_HOME', 'huggingface'), ('XDG_CACHE_HOME', 'xdg'),
                      ('VLLM_CACHE_ROOT', 'vllm'), ('TRITON_CACHE_DIR', 'triton'),
                      ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('CUDA_CACHE_PATH', 'cuda'),
                      ('TORCH_EXTENSIONS_DIR', 'torch_extensions'), ('TMPDIR', 'tmp')]:
        directory = cache/name
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    # Loopback requests must never traverse a desktop/system proxy.
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY', ''), '127.0.0.1', 'localhost', '::1']))
    env['no_proxy'] = env['NO_PROXY']
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    env.setdefault('OMP_NUM_THREADS', '8')
    return env


def server_command(profile, *, port=8010, gpu_fraction=None, max_model_len=8192):
    if profile not in PROFILES:
        raise ValueError(f'Unknown local VLM profile: {profile}')
    gpu_fraction = PROFILES[profile]['gpu_fraction'] if gpu_fraction is None else gpu_fraction
    if not 0 < gpu_fraction < 1 or max_model_len < 1024 or not 1 <= port <= 65535:
        raise ValueError('Invalid local VLM resource settings')
    python = Path(os.environ.get('ARM_LOCAL_VLM_PYTHON', ROOT/'.venv-vlm-local/bin/python'))
    path = model_root()/profile
    if not python.is_file() or not (path/'config.json').is_file():
        raise FileNotFoundError('Prepare the isolated VLM environment and pinned model first; see docs/local_vlm.txt')
    index = json.loads((path/'model.safetensors.index.json').read_text())
    if not all((path/name).is_file() for name in set(index['weight_map'].values())):
        raise FileNotFoundError('Model download is incomplete; finish the pinned download before serving')
    return [str(python), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(path), '--served-model-name', profile, '--host', '127.0.0.1',
        '--port', str(port), '--dtype', 'bfloat16', '--gpu-memory-utilization', str(gpu_fraction),
        '--max-model-len', str(max_model_len), '--max-num-seqs', '1',
        '--max-num-batched-tokens', '2048',
        '--limit-mm-per-prompt', '{"image":4,"video":0}',
        '--mm-processor-kwargs', '{"max_pixels":524288}',
        '--generation-config', 'vllm', '--seed', '0', '--enforce-eager']


def running_models(port=8010):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
            return [row['id'] for row in json.load(response)['data']]
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return []


def start_server(profile, *, port=8010, gpu_fraction=None, max_model_len=8192, timeout=600):
    active = running_models(port)
    if active:
        if profile not in active:
            raise RuntimeError(f'Port {port} already serves another model: {active}')
        return None
    directory = ROOT/'runs/local_vlm'
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory/f'{profile}.log'
    command = server_command(profile, port=port, gpu_fraction=gpu_fraction, max_model_len=max_model_len)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=service_environment(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (directory/f'server-{port}.json').write_text(json.dumps(dict(profile=profile,
        pid=process.pid, port=port, command=command, log_path=str(log_path)), indent=2))
    try:
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Local VLM exited ({process.returncode}); inspect {log_path}')
            if profile in running_models(port):
                return process
            time.sleep(1)
        raise TimeoutError(f'Local VLM startup timed out; inspect {log_path}')
    except BaseException:
        stop_process(process)
        raise


def stop_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def stop_server(port=8010):
    record = ROOT/f'runs/local_vlm/server-{port}.json'
    if not record.is_file():
        return False
    data = json.loads(record.read_text())
    cmdline = Path(f'/proc/{data["pid"]}/cmdline')
    if not cmdline.exists():
        return False
    command = cmdline.read_bytes().decode().split('\0')
    if ('vllm.entrypoints.openai.api_server' not in command or
            '--port' not in command or command[command.index('--port')+1] != str(port)):
        raise RuntimeError('Saved PID no longer identifies this local VLM service')
    os.killpg(data['pid'], signal.SIGTERM)
    return True


def apply_local_endpoint(args):
    profile = getattr(args, 'local_vlm', None)
    if not profile:
        return
    if profile not in PROFILES:
        raise ValueError(f'Unknown local VLM profile: {profile}')
    base = f'http://127.0.0.1:{getattr(args, "local_vlm_port", 8010)}/v1'
    # Override role-specific legacy environment defaults as well as the common API.
    for prefix, model_field in [('api', 'model'), ('vlm', 'model'), ('planner_llm', 'model'),
                               ('fallback_vlm', 'model'), ('grasp_vdm', 'model'), ('vlm_occlusion', 'model')]:
        setattr(args, prefix+'_base_url', base)
        setattr(args, prefix+'_api_key' if prefix != 'api' else 'api_key', 'local')
        setattr(args, prefix+'_'+model_field, profile)


def download(profile):
    from huggingface_hub import snapshot_download
    spec = PROFILES[profile]
    directory = model_root()/profile
    snapshot_download(repo_id=spec['repo'], revision=spec['revision'], local_dir=directory, token=False,
        max_workers=3, allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.txt', '*.model'])
    (directory/'armcode_model_revision.json').write_text(json.dumps(spec, indent=2))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['download', 'serve', 'status', 'stop'])
    parser.add_argument('--model', choices=tuple(PROFILES), default='qwen3vl-8b-awq')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--gpu-fraction', type=float)
    parser.add_argument('--max-model-len', type=int, default=8192)
    args = parser.parse_args()
    if args.operation == 'download':
        print(download(args.model))
    elif args.operation == 'status':
        print(json.dumps(dict(models=running_models(args.port)), ensure_ascii=False))
    elif args.operation == 'stop':
        print(json.dumps(dict(stop_requested=stop_server(args.port))))
    else:
        command = server_command(args.model, port=args.port, gpu_fraction=args.gpu_fraction,
                                 max_model_len=args.max_model_len)
        os.execvpe(command[0], command, service_environment())


if __name__ == '__main__':
    main()
