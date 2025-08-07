#!/usr/bin/env python3
"""
bench_all.py - one-stop benchmark & plotting driver for the OBDD project
-----------------------------------------------------------------------
This script *replaces* the previous helpers:
    * scripts/bench_runner.py
    * scripts/run_speed.sh
    * scripts/plot_results.py
    * scripts/plot_speed.py
    * scripts/plot_speedup.py
    * scripts/plot_stress.py

It can run both the micro benchmarks (easy / medium / worst‑case circuits)
and the Parity-n stress sweep, collect all CSV rows, and emit the PNG plots
in one shot.

HOW TO USE
==========
    python3 scripts/bench_all.py \
        --run-bench ./run_bench            # executable for micro cases
        --run-stress ./run_stress          # executable for Parity sweep
        --out micro_results.csv            # CSV for micro cases
        --stress-out stress_results.csv    # CSV for Parity sweep
        --bits 8 16                        # override for adder/equality
        --threads 1 2 4 8 12 16 20         # OpenMP thread counts
        --min 10 --max 24 --step 2 --rep 20# Parity sweep params

If you omit --run-stress the Parity sweep is skipped; same for --run-bench.

Dependencies: pandas >=1.5, matplotlib >=3.8.
"""

import argparse
import csv
import itertools
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# --------------------- CLI parsing ---------------------------
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run micro & stress benchmarks and generate plots.")

    p.add_argument("--run-bench", help="path to compiled run_bench executable")
    p.add_argument("--run-stress", help="path to compiled run_stress executable")
    p.add_argument("--run-sort", help="path to executable testing var-ordering")

    p.add_argument("--out", default="micro_results.csv",
                   help="output CSV for micro benchmarks")
    p.add_argument("--stress-out", default="stress_results.csv",
                   help="output CSV for Parity sweep")

    p.add_argument("--threads", nargs="*", type=int, default=[1, 2, 4, 8],
                   help="OpenMP thread counts to test (default: 1 2 4 8)")

    p.add_argument("--min", type=int, default=10, help="min parity bits (stress)")
    p.add_argument("--max", type=int, default=24, help="max parity bits (stress)")
    p.add_argument("--step", type=int, default=2, help="step parity bits (stress)")
    p.add_argument("--rep", type=int, default=5, help="repeat count (stress)")

    # override default BITS for some functions
    p.add_argument("--bits", nargs="*", type=int,
                   help="override default bit-sizes for adder/equality")

    p.add_argument("--no-plots", action="store_true",
                   help="skip PNG generation (CSV only)")

    return p.parse_args()

# ------------------------------------------------------------
# --------------------- MICRO bench ---------------------------
# ------------------------------------------------------------

FUNCS = ["majority", "mux", "adder", "equality", "parity"]
DEFAULT_BITS = {
    "majority": 3,
    "mux": 3,
    "adder": [8, 16],
    "equality": [8, 16],
    "parity": [16, 20],
}


def run_micro(exec_path: str, out_csv: pathlib.Path, threads: List[int], bits_override=None):
    if not exec_path:
        print("[micro] skipped - no --run-bench given")
        return

    if out_csv.exists():
        out_csv.unlink()

    bits_cfg = DEFAULT_BITS.copy()
    if bits_override:
        bits_cfg["adder"] = bits_cfg["equality"] = bits_override

    for func in FUNCS:
        bitlist = bits_cfg[func] if isinstance(bits_cfg[func], list) else [bits_cfg[func]]
        for b in bitlist:
            for thr in threads:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(thr)
                cmd = [exec_path, "--func", func, "--bits", str(b),
                       "--threads", str(thr), "--repeat", "3",
                       "--csvout", str(out_csv)]
                print("[micro]", " ".join(cmd))
                subprocess.check_call(cmd, env=env)
    print(f"[micro] OK results -> {out_csv}")


def run_sort(exec_path: str):
    if not exec_path:
        return
    print("[sort]", exec_path)
    subprocess.check_call([exec_path])

# ------------------------------------------------------------
# --------------------- STRESS Parity-n -----------------------
# ------------------------------------------------------------

def run_stress(exec_path: str, out_csv: pathlib.Path, minN: int, maxN: int, step: int, reps: int, threads: List[int]):
    if not exec_path:
        print("[stress] skipped - no --run-stress given")
        return

    if out_csv.exists():
        out_csv.unlink()
    header_written = False

    for n in range(minN, maxN + 1, step):
        for thr in threads:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(thr)

            cmd = [exec_path,
                   "--min", str(n), "--max", str(n), "--step", "2",
                   "--rep", str(reps), "--csv", "/dev/stdout"]
            print("[stress]", " ".join(cmd))
            out = subprocess.check_output(cmd, env=env, text=True)
            lines = [ln for ln in out.splitlines() if ln.startswith("Parity")]
            with out_csv.open("a") as f:
                if not header_written:
                    f.write("func,vars,threads,rep,nodes,ms\n")
                    header_written = True
                for ln in lines:
                    f.write(ln + "\n")
    print(f"[stress] OK results -> {out_csv}")

# ------------------------------------------------------------
# --------------------- PLOT helpers --------------------------
# ------------------------------------------------------------

def plot_micro(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)

    if {"name", "vars", "nodes", "ms"}.issubset(df.columns):
        df = df.rename(columns={"name": "func", "vars": "bits", "ms": "time_ms"})
        df["threads"] = 1
    elif {"func", "bits", "threads", "ms"}.issubset(df.columns):
        df = df.rename(columns={"ms": "time_ms"})
    else:
        raise ValueError("Micro CSV format not recognised")

    df_group = df.groupby(["func", "bits"]).mean(numeric_only=True).reset_index()

    plt.figure(figsize=(6, 4))
    for func, sub in df_group.groupby("func"):
        plt.plot(sub["bits"], sub["time_ms"], marker="o", label=func)
    plt.xlabel("Numero variabili")
    plt.ylabel("Tempo build (ms, medio)")
    plt.title("OBDD - tempo di costruzione")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("build_time.png")
    plt.close()
    print("[plot] build_time.png saved")

    if "nodes" in df.columns:
        plt.figure(figsize=(6, 4))
        for func, sub in df_group.groupby("func"):
            plt.plot(sub["bits"], sub["nodes"], marker="o", label=func)
        plt.xlabel("Numero variabili")
        plt.ylabel("Nodi nel BDD")
        plt.yscale("log")
        plt.title("OBDD - dimensione del grafo")
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig("nodes.png")
        plt.close()
        print("[plot] nodes.png saved")


def plot_stress(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    df = df[df["func"].str.startswith("Parity")]
    g = df.groupby(["vars", "threads"]).agg(ms_mean=("ms", "mean"), ms_std=("ms", "std"), nodes=("nodes", "mean")).reset_index()

    t1 = g[g.threads == g.threads.min()].set_index("vars")["ms_mean"]
    tn = g[g.threads == g.threads.max()].set_index("vars")["ms_mean"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(t1.index, t1.values, yerr=g[g.threads == g.threads.min()]["ms_std"], marker="o", label=f"{t1.name}-thr")
    plt.errorbar(tn.index, tn.values, yerr=g[g.threads == g.threads.max()]["ms_std"], marker="o", label=f"{tn.name}-thr")
    plt.xlabel("Variabili n")
    plt.ylabel("Tempo medio [ms]")
    plt.title("Tempo build Parity(n) - 1 vs N thread")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_vs_vars.png")
    plt.close()
    print("[plot] time_vs_vars.png saved")

    # ---- speed-up -------------------------------------------------
    speedup = t1 / tn
    plt.figure(figsize=(6, 4))
    plt.plot(speedup.index, speedup.values, "s-")
    plt.xlabel("Variabili n")
    plt.ylabel("Speed-up T₁ / Tₙ")
    plt.title("Speed-up OpenMP (Parity)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speedup.png")
    plt.close()
    print("[plot] speedup.png saved")

    # ---- nodi -----------------------------------------------------
    if "nodes" in g.columns:
        nodes = g[g.threads == g.threads.min()].set_index("vars")["nodes"]
        plt.figure(figsize=(6, 4))
        plt.plot(nodes.index, nodes.values, "d-")
        plt.xlabel("Variabili n")
        plt.ylabel("#Nodi (media)")
        plt.yscale("log")
        plt.title("Crescita nodi Parity (atteso 2^n − 1)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("nodes_vs_vars.png")
        plt.close()
        print("[plot] nodes_vs_vars.png saved")


# ------------------------------------------------------------
# --------------------- ENTRY-POINT ---------------------------
# ------------------------------------------------------------

def main():
    args = parse_args()
    threads = args.threads

    # micro benchmarks
    if args.run_bench:
        run_micro(args.run_bench, pathlib.Path(args.out), threads, args.bits)
        if not args.no_plots:
            plot_micro(pathlib.Path(args.out))

    # stress benchmarks
    if args.run_stress:
        run_stress(args.run_stress, pathlib.Path(args.stress_out),
                   args.min, args.max, args.step, args.rep, threads)
        if not args.no_plots:
            plot_stress(pathlib.Path(args.stress_out))

    # optional sort benchmark
    if args.run_sort:
        run_sort(args.run_sort)

    print("✔ benchmark(s) completed on",
          datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()

