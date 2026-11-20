"""Distribute independent committed clause validations without changing their order."""
from concurrent.futures import ProcessPoolExecutor as LocalPool
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

MEASURES = Path(__file__).resolve().parents[1] / 'revision/measures'
sys.path.insert(0, str(MEASURES))
import run_measures as frozen


def partition(tasks):
    return list(enumerate(tasks))[::2], list(enumerate(tasks))[1::2]


def merge(total, local, remote):
    rows = local + remote
    indices = [index for index, _ in rows]
    if len(indices) != total or set(indices) != set(range(total)):
        raise ValueError('Distributed validation coverage differs')
    return [value for _, value in sorted(rows)]


def ssh(command, data=None):
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=12',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
            '-i', os.environ['CLAUSE_PEER_KEY'], '-p', os.environ['CLAUSE_PEER_PORT'],
            os.environ['CLAUSE_PEER_USER'] + '@' + os.environ['CLAUSE_PEER_HOST'],
            'bash -lc ' + shlex.quote(command)]
    for attempt in range(4):
        response = subprocess.run(argv, input=data, capture_output=True, timeout=180)
        if response.returncode == 0:
            return response.stdout
        if attempt == 3:
            raise RuntimeError('Peer command failed: ' + response.stderr.decode(errors='replace')[-600:])
        time.sleep(4 * (attempt + 1))


class SharedPool:
    def __init__(self, max_workers):
        self.workers = max_workers
        self.local = LocalPool(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.local.shutdown(wait=True, cancel_futures=error[0] is not None)

    def map(self, function, tasks, chunksize=1):
        if function is not frozen.validate_shape:
            raise ValueError('Only independent frozen shape validation may be distributed')
        tasks = list(tasks)
        local, remote = partition(tasks)
        work = Path(os.environ.get('CLAUSE_SHARED_WORK', '/workspace/clause-shared'))
        work.mkdir(parents=True, exist_ok=False)
        payload = json.dumps(remote).encode()
        input_sha = hashlib.sha256(payload).hexdigest()
        ssh('mkdir -p /workspace/clause-peer; rm -f /workspace/clause-peer/worker.exit /workspace/clause-peer/results.json; cat > /workspace/clause-peer/tasks.json', payload)
        script = ('set -euo pipefail\ncd /workspace\n'
                  "trap 'echo $? > /workspace/clause-peer/worker.exit' EXIT\n"
                  f'/workspace/venv/bin/python /workspace/post_reg/run/clause_parallel.py --worker '
                  f'--input /workspace/clause-peer/tasks.json --input-sha {input_sha} '
                  f'--output /workspace/clause-peer/results.json --workers {self.workers}\n')
        ssh('cat > /workspace/clause-peer/run.sh', script.encode())
        ssh('nohup setsid bash /workspace/clause-peer/run.sh > /workspace/clause-peer/worker.log 2>&1 </dev/null &')
        local_values = list(self.local.map(function, [task for _, task in local], chunksize=chunksize))
        while True:
            status = ssh('test ! -f /workspace/clause-peer/worker.exit || cat /workspace/clause-peer/worker.exit').strip()
            if status:
                if status != b'0':
                    raise RuntimeError('Peer validation failed; inspect /workspace/clause-peer/worker.log')
                break
            time.sleep(20)
        remote_bytes = ssh('cat /workspace/clause-peer/results.json')
        remote_sha = ssh('sha256sum /workspace/clause-peer/results.json').decode().split()[0]
        if hashlib.sha256(remote_bytes).hexdigest() != remote_sha:
            raise ValueError('Peer result checksum differs')
        (work/'peer_results.json').write_bytes(remote_bytes)
        (work/'assignment.json').write_text(json.dumps(dict(
            local_indices=[i for i, _ in local], remote_indices=[i for i, _ in remote],
            tasks=len(tasks), peer_result_sha256=remote_sha,
            method='Alternating task indices; original validation function and restored original order'))+'\n')
        local_rows = [[index, value] for (index, _), value in zip(local, local_values, strict=True)]
        return iter(merge(len(tasks), local_rows, json.loads(remote_bytes)))


def worker(args):
    frozen.verify_inputs(MEASURES.parents[1], Path('/workspace/run/inputs/source-passages.json'))
    data = args.input.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.input_sha:
        raise ValueError('Peer task input checksum differs')
    frozen.verify_inputs(MEASURES.parents[1], Path('/workspace/run/inputs/source-passages.json'))
    indexed = json.loads(data)
    with LocalPool(max_workers=args.workers) as pool:
        values = list(pool.map(frozen.validate_shape, [task for _, task in indexed], chunksize=8))
    rows = [[index, value] for (index, _), value in zip(indexed, values, strict=True)]
    args.output.write_text(json.dumps(rows)+'\n')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--worker', action='store_true')
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--input-sha', required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--workers', type=int, default=8)
        worker(parser.parse_args())
    else:
        import concurrent.futures
        concurrent.futures.ProcessPoolExecutor = SharedPool
        # Importing the sealed module avoids executing its duplicate CLI guards.
        frozen.main()
