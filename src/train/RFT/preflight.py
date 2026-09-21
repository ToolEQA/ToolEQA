"""Fail-fast checks for the full ToolEQA online RFT stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from src.train.RFT.dataset import semantic_task_key
from src.train.RFT.evidence import evidence_target_issues, resolve_evidence_targets


def process_is_running(script_names: tuple[str, ...]) -> bool:
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = [part.decode(errors="ignore") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not parts or not Path(parts[0]).name.startswith("python"):
            continue
        if any(Path(part).name in script_names for part in parts[1:]):
            return True
    return False


def audit_jsonl(path: Path, label: str, official_test: bool = False) -> list[str]:
    """Validate every production row before launching expensive workers."""
    failures: list[str] = []
    required = {"data_source", "prompt", "reward_model", "agent_name", "extra_info"}
    seen_tasks: dict[str, int] = {}
    invalid_rows: list[str] = []
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            record = json.loads(line)
            missing = required - record.keys()
            if missing:
                invalid_rows.append(f"line {line_number} missing keys {sorted(missing)}")
                continue
            if record.get("agent_name") != "tooleqa_evidence_agent":
                invalid_rows.append(f"line {line_number} has wrong agent_name")
            info = record.get("extra_info") or {}
            if official_test:
                if info.get("official_test_split") not in {"seen", "unseen"}:
                    invalid_rows.append(f"line {line_number} lacks official split marker")
                if not info.get("scene") or not info.get("question") or not info.get("answer"):
                    invalid_rows.append(f"line {line_number} lacks test metadata")
                continue
            issues = list(info.get("reward_audit") or [])
            issues.extend(evidence_target_issues(record))
            if info.get("reward_eligible") is not True:
                issues.append("missing-or-false-reward-eligible")
            if not info.get("evidence_targets"):
                issues.append("missing-evidence-targets")
            try:
                resolve_evidence_targets(info, require_explicit=True)
            except ValueError as error:
                issues.append(f"invalid-audit-fields:{error}")
            task_key = semantic_task_key(record)
            if task_key in seen_tasks:
                issues.append(f"semantic-duplicate:line-{seen_tasks[task_key]}")
            else:
                seen_tasks[task_key] = line_number
            if issues:
                invalid_rows.append(f"line {line_number}: {list(dict.fromkeys(issues))}")
    if row_count == 0:
        failures.append(f"{label} JSONL is empty: {path}")
    if invalid_rows:
        preview = "; ".join(invalid_rows[:5])
        failures.append(f"{label} JSONL has {len(invalid_rows)} reward-unsafe rows ({preview})")
    return failures


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--train", default=str(root / "src/train/RFT/data/train_reward_eligible.jsonl")
    )
    parser.add_argument(
        "--validation", default=str(root / "src/train/RFT/data/validation_reward_eligible.jsonl")
    )
    parser.add_argument("--channel", type=int, default=int(os.environ.get("TOOLEQA_TOOL_GPU_ID", "0")))
    parser.add_argument("--require-detany", action="store_true")
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--official-test", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    model = Path(args.model).expanduser().resolve()
    if not (model / "config.json").is_file():
        failures.append(f"model is not a Hugging Face checkpoint: {model}")
    weight_files = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
    if model.is_dir() and not weight_files:
        failures.append(f"model has no merged weights: {model} (merge a LoRA adapter before vLLM)")

    for label, raw_path in (("train", args.train), ("validation", args.validation)):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            failures.append(f"{label} JSONL is missing: {path}")
            continue
        try:
            failures.extend(audit_jsonl(path, label, args.official_test and label == "validation"))
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(f"cannot read {label} JSONL: {error}")

    static_paths = (
        root / "third_party/verl/verl",
        root / "third_party/DetAny3D/app_mp.py",
        root / "data/ToolTrajectory/prompts/rft_thought_code_system_prompt.txt",
        root / "config/react-eqa.yaml",
    )
    for path in static_paths:
        if not path.exists():
            failures.append(f"required path is missing: {path}")

    if not args.skip_scenes:
        scene_paths = (root / "data/HM3D", root / "data/OpenEQA/scenes")
        if os.environ.get("TOOLEQA_SCENE_ROOT"):
            scene_paths = (Path(os.environ["TOOLEQA_SCENE_ROOT"]),)
        existing_scene_paths = [path for path in scene_paths if path.is_dir()]
        if args.official_test and Path(args.validation).is_file():
            missing_test_scenes = set()
            with Path(args.validation).open() as handle:
                for line in handle:
                    scene = json.loads(line)["extra_info"]["scene"]
                    if not any(
                        (base / scene / f"{scene[6:]}.basis.glb").is_file()
                        and (base / scene / f"{scene[6:]}.basis.navmesh").is_file()
                        for base in scene_paths
                    ):
                        missing_test_scenes.add(scene)
            if missing_test_scenes:
                failures.append(f"official test scenes missing: {len(missing_test_scenes)}; "
                                f"examples: {sorted(missing_test_scenes)[:5]}")
        for path in scene_paths:
            if not path.is_dir():
                warnings.append(f"scene root is absent: {path}")
        if not existing_scene_paths:
            failures.append("no configured Habitat scene root exists")
        else:
            unresolved: set[str] = set()
            train_path = Path(args.train).expanduser().resolve()
            if train_path.is_file():
                with train_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle):
                        if line_number >= 1000:
                            break
                        scene = (json.loads(line).get("extra_info") or {}).get("scene")
                        if scene and not any((path / scene).is_dir() for path in existing_scene_paths):
                            unresolved.add(scene)
            if unresolved:
                examples = ", ".join(sorted(unresolved)[:5])
                failures.append(f"training scenes are unresolved ({len(unresolved)} in first 1000 records): {examples}")

    missing_packages = [
        name
        for name in ("torch", "ray", "vllm", "verl", "habitat_sim", "posix_ipc", "omegaconf")
        if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        failures.append(f"Python environment misses packages: {', '.join(missing_packages)}")

    ipc_paths = [Path(f"/dev/shm/image_data_{args.channel}"), Path(f"/dev/shm/result_data_{args.channel}")]
    ipc_ready = all(path.exists() for path in ipc_paths)
    server_running = process_is_running(("detany_server.py", "app_mp.py"))
    if args.require_detany and (not ipc_ready or not server_running):
        failures.append(
            f"DetAny3D channel {args.channel} is not live; run scripts/run_detany3d.sh on the reserved GPU"
        )
    elif not ipc_ready or not server_running:
        warnings.append(f"DetAny3D channel {args.channel} is not currently live")

    print(f"model: {model}")
    print(f"DetAny3D channel {args.channel}: ipc={'yes' if ipc_ready else 'no'}, process={'yes' if server_running else 'no'}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        raise SystemExit(1)
    print("preflight: OK")


if __name__ == "__main__":
    main()
