"""Extract measured full-step timings and audit saved speed-probe trajectories."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean

from src.train.RFT.resume_official import hardware_errors

FIELDS = ("step", "timing_s/step", "timing_s/generate_async", "timing_s/old_log_prob",
          "timing_s/ref", "timing_s/update_actor", "timing_s/update_weights",
          "actor/grad_norm", "training/rollout_actor_probs_pearson_corr")


def summarize(log):
    steps = []
    for line in log.read_text().splitlines():
        if " - training/global_step:" not in line:
            continue
        fields = {k: float(v) for k, v in re.findall(
            r"([\w/]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", line)}
        steps.append({k: fields[k] for k in FIELDS if k in fields})
    steady = steps[1:]
    result = {"log": str(log), "steps": steps,
              "post_first_step_mean_seconds": fmean(s["timing_s/step"] for s in steady) if steady else None}
    run = log.with_suffix("")
    traces = [json.loads(p.read_text()) for p in (run / "trajectories").glob("step_*/*.json")]
    result["trajectory_count"] = len(traces)
    result["trajectories_per_step"] = dict(Counter(r.get("global_step") for r in traces))
    result["hardware_error_count"] = sum(bool(hardware_errors(r.get("tool_trace", []))) for r in traces)
    result["missing_reward_count"] = sum("reward" not in r for r in traces)
    result["option_leakage_count"] = sum("Choices:" in r.get("task", "") for r in traces)
    initial_images = [r.get("initial_image") for r in traces]
    result["duplicate_initial_image_paths"] = len(initial_images) - len(set(initial_images))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(log) for log in args.logs], indent=2))
