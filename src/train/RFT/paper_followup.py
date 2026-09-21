"""One-shot completion gate resuming this conversation after the pilot ends."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def readiness(run_root):
    counts = {}
    for split, expected in (("seen", 845), ("unseen", 1069)):
        summary = run_root / f"{split}-summary.json"
        if not summary.is_file():
            return False, f"waiting for {split} summary"
        try:
            data = json.loads(summary.read_text())
        except json.JSONDecodeError:
            return False, f"waiting for complete {split} summary write"
        if not isinstance(data, list) or len(data) != 1 or data[0].get("records") != expected:
            raise ValueError(f"{split} summary does not contain {expected} records")
        with (run_root / "data-mounted" / f"{split}.jsonl").open() as handle:
            expected_ids = [json.loads(line)["extra_info"]["sample_id"] for line in handle]
        observed = []
        for file in (run_root / split / "trajectories").glob("step_*/*.json"):
            record = json.loads(file.read_text())
            if "acc" not in (record.get("reward") or {}):
                return False, f"waiting for {split} scoring"
            observed.append(record["sample"]["sample_id"])
        if len(observed) != expected or len(set(observed)) != expected:
            raise ValueError(f"{split} has incomplete or duplicate trajectory IDs")
        if set(observed) != set(expected_ids):
            raise ValueError(f"{split} trajectory IDs differ from the official manifest")
        counts[split] = expected
    return True, counts


def launcher_alive(run_root):
    try:
        pid = int((run_root / "evaluation.pid").read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return b"run_step0_official_tests.sh" in cmd


def write_status(path, **fields):
    payload = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    state = args.run_root / "paper-followup"
    if args.check_only:
        ready, detail = readiness(args.run_root)
        print(json.dumps({"ready": ready, "detail": detail,
                          "launcher_alive": launcher_alive(args.run_root)}))
        return
    state.mkdir(exist_ok=True)
    lock = (state / "watcher.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = state / "status.json"
    if (state / "invoked.json").exists():
        raise SystemExit("This one-shot continuation has already been invoked")
    while True:
        try:
            ready, detail = readiness(args.run_root)
            alive = launcher_alive(args.run_root)
        except (ValueError, KeyError) as error:
            write_status(status_file, state="failed_validation", error=str(error))
            raise
        if ready and not alive:
            break
        if not alive and not ready:
            write_status(status_file, state="dependency_exited_incomplete", detail=detail)
            raise SystemExit("Pilot exited before both full evaluations completed")
        write_status(status_file, state="waiting", detail=detail,
                     watcher_pid=os.getpid(), session_id=args.session_id)
        time.sleep(30)
    prompt = (repo / "assets/iclr2027_followup_prompt.txt").read_text()
    command = ["/home/zml/.local/bin/codex", "exec", "--sandbox", "danger-full-access",
               "-c", 'approval_policy="never"', "resume", args.session_id,
               "--json", "--output-last-message", str(state / "last-message.md"), "-"]
    write_status(state / "invoked.json", state="invoking", session_id=args.session_id)
    write_status(status_file, state="resuming_conversation", session_id=args.session_id)
    with (state / "continuation.jsonl").open("a") as out, (state / "continuation.stderr.log").open("a") as err:
        result = subprocess.run(command, input=prompt, text=True, cwd=repo, stdout=out, stderr=err)
    write_status(status_file, state="continuation_returned" if result.returncode == 0
                 else "continuation_failed", returncode=result.returncode,
                 session_id=args.session_id)


if __name__ == "__main__":
    main()
