# VQC Target-State Preparation Benchmark

This repository is a compact showcase of practical VQC engineering:

- a parameterized 2-qubit variational circuit
- a single strict metric (`P(|11>)`) used as the only success criterion
- local budgeted hill-climb training with a random baseline
- optional repeated hardware proof with explicit success-rate reporting

The experiment is intentionally small and reproducible, but the flow is production-style: clean artifact JSON, fixed budgets, explicit thresholds, and clear success accounting.

## What is being optimized

We maximize the probability of the output bitstring `11`:

`P11 = <11| ρ |11>`

This is represented as a Pauli-expansion observable:

`P11 = 1/4 (II - ZI - IZ + ZZ)`

Strict success condition: `P11 > 0.75`.

## Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Local training benchmark

```bash
python3 vqc_benchmark.py --mode train
```

This writes `results/vqc_benchmark.json` with:

- budgeted hill-climb result (best parameters + objective trace)
- random baseline result
- aggregate best reference selection

## Hardware strict proof (single metric)

Use trained parameters from local benchmark and repeat hardware evaluations:

```bash
python3 vqc_benchmark.py \
  --mode proof \
  --params-file results/vqc_benchmark.json \
  --proof-repetitions 3
```

The proof payload includes:

- `run_summaries` with per-job `job_id`, `p11`, and per-run stdev
- `run_count`, `success_count`, `strict_success_rate`
- `p11`, `p11_ci95`, and circuit metadata

Success is strict and objective-only:
- `strict_success_rate = number_of_runs_with_P11>0.75 / run_count`

## Latest proof (2026-04-16 UTC)

The reproducible hardware witness was rerun after the Python/runtime refresh:

- Backend: `ibm_fez` (`open-instance`)
- Proof repetitions: `3`
- Best trained parameters used: from `results/vqc_benchmark.json`
- Hardware jobs:
  - `d7g2pvdp8b1s73arl0m0`
  - `d7g2q4dp8b1s73arl0sg`
  - `d7g2q8ua0v2s738abdg0`
- Metric: `|11>` probability
- Mean `P(|11>) = 0.978671733318607`
- `p11_ci95 = 0.0005117366127263978`
- Strict successes: `3 / 3` (`strict_success_rate = 1.0`)
- Metric-only pass criterion: `0.75`

## Why this is a stronger showcase than CHSH

- It adds a **small variational training loop** (optimization + budgeting)
- It compares against a classical random baseline
- It keeps the reporting metric single-purpose and objective-only
- It includes repeat hardware proof as reproducibility evidence, not a single run

## Project structure

```text
.
├── README.md
├── requirements.txt
├── Makefile
├── vqc_benchmark.py
├── results/
└── assets/
```

## Optional hardeners

- increase `--max-evals` and `--random-baseline-samples`
- pin `--backend`/`--instance`
- increase `--proof-repetitions`

## Note

If runtime auth is not configured, local mode still works and produces a complete local proof artifact.
