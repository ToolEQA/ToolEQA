"""Archive and validate a supervised probe when it exits; never start training."""
import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from src.train.RFT.supervise_run import write_status
from src.train.RFT.verify_memory_probe import inspect


def check_archive(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'Corrupt checkpoint archive: {path}: {bad}')
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--ray-logs', required=True, type=Path)
    args = parser.parse_args()
    result_path = args.directory / 'verification.json'
    try:
        while True:
            state = json.loads((args.directory / 'status.json').read_text())
            if state['status'] != 'running':
                break
            time.sleep(15)
        target = args.directory / 'worker-logs'
        target.mkdir(exist_ok=True)
        for path in args.ray_logs.glob('worker-*.out'):
            if 'TOOLEQA_MEMORY ' in path.read_text(errors='replace'):
                shutil.copy2(path, target / path.name)
        result = inspect(args.directory, args.run, target)
        if result['passed']:
            # Verify all tensor archive CRCs, not just presence and file size.
            with ThreadPoolExecutor(max_workers=3) as pool:
                checked = list(pool.map(check_archive, sorted((args.run / 'checkpoints').rglob('*.pt'))))
            result['checkpoint_archives_crc_passed'] = checked
        write_status(result_path, result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['passed'] else 1
    except Exception as error:
        write_status(result_path, {'passed': False, 'error': repr(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
