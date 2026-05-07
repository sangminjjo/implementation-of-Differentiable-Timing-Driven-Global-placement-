#!/usr/bin/env python3
"""
sweep_timing.py
Sweep timing_growth_factor over [1.005, ..., 1.015] for superblue1.
Runs each experiment sequentially inside the lucid_morse Docker container.
Results are saved to results/superblue1_factor{f}/ and
a summary CSV is written incrementally to results/timing_sweep_summary.csv.
"""

import os
import json
import subprocess
import csv
import re
import time

CONTAINER  = "lucid_morse"
WORKDIR    = "/DREAMPlace"
BASE_CFG   = "test/iccad2015.ot/superblue1.json"
HOST_ROOT  = "/DREAMPlace/"
FACTORS    = [round(1.004 + i * 0.001,3) for i in range(11)]
CSV_PATH   = os.path.join(HOST_ROOT, "results", "timing_sweep_summary.csv")


def run_experiment(factor: float) -> dict:
    f_str      = f"{factor:.3f}"
    result_sub = f"results/superblue1_factor{f_str}"
    cfg_name   = f"sweep_cfg_{f_str}.json"
    log_rel    = f"{result_sub}/sweep.log"
    log_host   = os.path.join(HOST_ROOT, log_rel)

    # Build per-experiment config
    with open(os.path.join(HOST_ROOT, BASE_CFG)) as fp:
        cfg = json.load(fp)
    cfg["timing_growth_factor"] = factor
    cfg["result_dir"]           = result_sub
    cfg["plot_flag"]             = 0  # skip plotting to save time

    cfg_host = os.path.join(HOST_ROOT, cfg_name)
    with open(cfg_host, "w") as fp:
        json.dump(cfg, fp, indent=2)

    os.makedirs(os.path.join(HOST_ROOT, result_sub), exist_ok=True)

    cmd = [
        "python", "dreamplace/Placer.py", cfg_name
    ]

    print(f"\n[factor={f_str}] starting ...", flush=True)
    t0  = time.time()
    with open(log_host, "w") as log_fp:
        ret = subprocess.run(cmd, cwd=WORKDIR, stdout=log_fp, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[factor={f_str}] done in {elapsed/60:.1f} min  (exit={ret.returncode})", flush=True)

    # Parse post-legalization WNS/TNS
    wns_val = tns_val = None
    try:
        with open(log_host) as fp:
            content = fp.read()

        # Grab only the text after the last "additional sta after legalization" marker
        parts = content.split("additional sta after legalization")
        search_region = parts[-1] if len(parts) > 1 else content

        # Log format:  tns : -1.234, wns: -0.056
        matches = re.findall(
            r'tns\s*:\s*([-\d.eE+]+),\s*wns:\s*([-\d.eE+]+)',
            search_region
        )
        if matches:
            tns_val = float(matches[0][0])
            wns_val = float(matches[0][1])
    except Exception as e:
        print(f"  [warn] log parse error: {e}", flush=True)

    if wns_val is not None:
        print(f"  post-legal WNS={wns_val:.4f} ns  TNS={tns_val:.4f} ns", flush=True)
    else:
        print(f"  [warn] could not extract WNS/TNS — check {log_host}", flush=True)

    return {
        "factor":      f_str,
        "wns_ns":      wns_val,
        "tns_ns":      tns_val,
        "elapsed_min": round(elapsed / 60, 1),
        "exit_code":   ret.returncode,
    }


def write_csv(rows: list):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["factor", "wns_ns", "tns_ns", "elapsed_min", "exit_code"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv] {CSV_PATH}", flush=True)


def main():
    print(f"Sweep factors: {FACTORS}", flush=True)
    summary = []

    for f in FACTORS:
        row = run_experiment(f)
        summary.append(row)
        write_csv(summary)   # incremental write so partial results survive a crash

    print("\n=== SWEEP COMPLETE ===")
    print(f"Summary: {CSV_PATH}")


if __name__ == "__main__":
    main()
