#!/usr/bin/env python3
"""VQC benchmark with strict metric-only success proof.

This project optimizes a small variational circuit to maximize the target-state
probability P(|11>) for two qubits, with optional hardware proof-by-repetition.
Success is purely defined as the mean P(|11>) exceeding a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService


TARGET_PROBABILITY = SparsePauliOp.from_list(
    [
        ("II", 0.25),
        ("ZI", -0.25),
        ("IZ", -0.25),
        ("ZZ", 0.25),
    ]
)

PARAM_COUNT = 8
METRIC_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small VQC training + strict metric-only success benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "proof"),
        default="train",
        help="train: local hill-climb benchmark, proof: repeated hardware witness check",
    )
    parser.add_argument(
        "--output",
        default="results/vqc_benchmark.json",
        help="Output JSON path for the primary result.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used for train initialization and sampling.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=64,
        help="Objective-call budget for budgeted local training.",
    )
    parser.add_argument(
        "--random-baseline-samples",
        type=int,
        default=32,
        help="Number of random-basis points for baseline.",
    )
    parser.add_argument(
        "--backend",
        default="ibm_fez",
        help="IBM backend for proof mode.",
    )
    parser.add_argument(
        "--instance",
        default="open-instance",
        help="IBM Quantum Runtime instance name.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Preset transpiler optimization level for proof mode.",
    )
    parser.add_argument(
        "--proof-repetitions",
        type=int,
        default=3,
        help="Number of hardware proof runs (proof mode only).",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="JSON file with best_params from train mode (proof mode only).",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Comma-separated 8 parameters to evaluate in proof mode.",
    )
    parser.add_argument(
        "--save-circuit",
        default=None,
        help="Optional path to save the logical circuit image.",
    )
    return parser.parse_args()


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return float(value)
    return float(value)


def aggregate(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    value_mean = mean(values)
    value_stdev = pstdev(values) if len(values) > 1 else 0.0
    value_ci95 = 0.0 if len(values) <= 1 else 1.96 * (value_stdev / math.sqrt(len(values)))
    return value_mean, value_stdev, value_ci95


def build_vqc(params: np.ndarray) -> QuantumCircuit:
    theta = np.asarray(params, dtype=float)
    if theta.shape != (PARAM_COUNT,):
        raise ValueError(f"Expected {PARAM_COUNT} parameters, got {theta.shape}")

    circuit = QuantumCircuit(2, name="vqc_target_11")
    circuit.ry(theta[0], 0)
    circuit.rz(theta[1], 0)
    circuit.ry(theta[2], 1)
    circuit.rz(theta[3], 1)
    circuit.cx(0, 1)
    circuit.ry(theta[4], 0)
    circuit.rz(theta[5], 0)
    circuit.ry(theta[6], 1)
    circuit.rz(theta[7], 1)
    return circuit


def evaluate_local(estimator: StatevectorEstimator, params: np.ndarray) -> float:
    circuit = build_vqc(params)
    result = estimator.run([(circuit, TARGET_PROBABILITY)]).result()[0]
    return float(np.clip(to_float(result.data.evs), 0.0, 1.0))


def budgeted_hill_climb(
    objective: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    max_evals: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    max_evals = max(1, int(max_evals))
    params = rng.uniform(0.0, 2.0 * math.pi, size=PARAM_COUNT)
    value = objective(params)

    history: list[dict[str, float]] = [
        {
            "eval": 1,
            "source": "init",
            "p11": value,
            "params": params.tolist(),
        }
    ]

    eval_count = 1
    best_params = params
    best_value = value
    step = 0.6

    while eval_count < max_evals:
        improved = False
        for index in range(PARAM_COUNT):
            for direction in (-1.0, 1.0):
                if eval_count >= max_evals:
                    break
                candidate = np.array(best_params, copy=True)
                candidate[index] = (candidate[index] + direction * step) % (2.0 * math.pi)
                candidate_value = objective(candidate)
                eval_count += 1
                history.append(
                    {
                        "eval": eval_count,
                        "source": f"step_{index}_{'plus' if direction > 0 else 'minus'}",
                        "p11": candidate_value,
                        "params": candidate.tolist(),
                    }
                )
                if candidate_value > best_value:
                    best_value = candidate_value
                    best_params = candidate
                    improved = True
                    break
            if improved:
                break

        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    return best_params, best_value, history


def run_random_baseline(
    objective: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    samples: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    samples = max(0, int(samples))
    history = []
    best_value = -math.inf
    best_params = np.zeros(PARAM_COUNT)

    for sample_index in range(1, samples + 1):
        params = rng.uniform(0.0, 2.0 * math.pi, size=PARAM_COUNT)
        value = objective(params)
        history.append(
            {
                "sample": sample_index,
                "p11": value,
                "params": params.tolist(),
            }
        )
        if value > best_value:
            best_value = value
            best_params = params

    return best_params, best_value, history


def base_result(mode: str) -> dict[str, Any]:
    return {
        "experiment": "vqc_target_state_preparation",
        "metric": "P(|11>)",
        "metric_threshold": METRIC_THRESHOLD,
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "param_count": PARAM_COUNT,
        "objective_observable": {
            "label": "P(|11>)",
            "terms": [
                {"pauli": "II", "coefficient": 0.25},
                {"pauli": "ZI", "coefficient": -0.25},
                {"pauli": "IZ", "coefficient": -0.25},
                {"pauli": "ZZ", "coefficient": 0.25},
            ],
        },
    }


def summarize_training_result(
    seed: int,
    rng: np.random.Generator,
    max_evals: int,
    random_samples: int,
    estimator: StatevectorEstimator,
) -> dict[str, Any]:
    objective = lambda params: evaluate_local(estimator, params)

    best_hc_params, best_hc_value, hc_history = budgeted_hill_climb(
        objective=objective,
        rng=rng,
        max_evals=max_evals,
    )

    rb_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    rb_params, rb_value, rb_history = run_random_baseline(
        objective=objective,
        rng=rb_rng,
        samples=random_samples,
    )

    return {
        "seed": int(seed),
        "hill_climb": {
            "best_p11": best_hc_value,
            "best_params": best_hc_params.tolist(),
            "strict_success": best_hc_value >= METRIC_THRESHOLD,
            "eval_budget": len(hc_history),
            "history": hc_history,
            "history_size": len(hc_history),
        },
        "random_baseline": {
            "best_p11": rb_value,
            "best_params": rb_params.tolist(),
            "strict_success": rb_value >= METRIC_THRESHOLD,
            "sample_count": random_samples,
            "history": rb_history,
            "history_size": len(rb_history),
        },
    }


def run_proof_hardware(
    params: list[float],
    backend_name: str,
    instance: str,
    optimization_level: int,
    repetitions: int,
    save_circuit_path: str | None = None,
) -> dict[str, Any]:
    repetitions = max(1, int(repetitions))
    float_params = np.asarray(params, dtype=float)
    if float_params.shape != (PARAM_COUNT,):
        raise ValueError(f"Expected {PARAM_COUNT} params, got shape {float_params.shape}")

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)

    base_template = build_vqc(np.zeros(PARAM_COUNT))
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )
    transpiled_template = pass_manager.run(base_template)
    transpiled_observable = TARGET_PROBABILITY.apply_layout(transpiled_template.layout)

    if save_circuit_path:
        path = Path(save_circuit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        transpiled_template.draw("mpl", idle_wires=False, style="iqp").savefig(path)

    estimator = Estimator(mode=backend)

    run_summaries: list[dict[str, Any]] = []
    p11_values: list[float] = []
    std_values: list[float] = []

    for run_index in range(1, repetitions + 1):
        circuit = build_vqc(float_params)
        transpiled_circuit = pass_manager.run(circuit)
        job = estimator.run([(transpiled_circuit, transpiled_observable)])
        result = job.result()[0]
        p11 = to_float(result.data.evs)
        p11 = float(np.clip(p11, 0.0, 1.0))
        std = to_float(result.data.stds)
        p11_values.append(p11)
        std_values.append(std)
        run_summaries.append(
            {
                "run_index": run_index,
                "job_id": job.job_id(),
                "p11": p11,
                "std": std,
                "strict_success": p11 >= METRIC_THRESHOLD,
            }
        )

    mean_p11, p11_stdev, p11_ci95 = aggregate(p11_values)
    mean_std, _, _ = aggregate(std_values)
    success_count = sum(1 for entry in run_summaries if entry["strict_success"])

    return {
        "backend": backend.name,
        "instance": instance,
        "job_count": repetitions,
        "job_id_first": run_summaries[0]["job_id"],
        "transpiled_depth": transpiled_template.depth(),
        "two_qubit_gate_count": sum(
            1 for instruction in transpiled_template.data if instruction.operation.num_qubits == 2
        ),
        "p11": mean_p11,
        "p11_std": p11_stdev,
        "p11_ci95": p11_ci95,
        "estimator_std_mean": mean_std,
        "strict_success_rate": success_count / repetitions,
        "success_count": success_count,
        "run_summaries": run_summaries,
        "abs_p11_values": [abs(v) for v in p11_values],
        "run_count": repetitions,
        "violates_metric": mean_p11 >= METRIC_THRESHOLD,
    }


def resolve_params_from_args(args: argparse.Namespace) -> list[float]:
    if args.params:
        values = [float(value) for value in args.params.split(",")]
        if len(values) != PARAM_COUNT:
            raise ValueError(f"--params must contain exactly {PARAM_COUNT} values")
        return values

    if args.params_file:
        path = Path(args.params_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "hill_climb" in payload and "best_params" in payload["hill_climb"]:
            return list(payload["hill_climb"]["best_params"])
        if "best_params" in payload:
            return list(payload["best_params"])
        raise KeyError("Could not find best_params in params file")

    raise ValueError("proof mode requires --params or --params-file")


def save_circuit_if_requested(path: str | None, params: np.ndarray) -> None:
    if not path:
        return
    circuit = build_vqc(params)
    figure = circuit.draw("mpl", idle_wires=False, style="iqp")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight")
    figure.clf()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        rng = np.random.default_rng(args.seed)
        estimator = StatevectorEstimator()
        train_payload = summarize_training_result(
            seed=args.seed,
            rng=rng,
            max_evals=args.max_evals,
            random_samples=args.random_baseline_samples,
            estimator=estimator,
        )

        best_hc_params = np.array(train_payload["hill_climb"]["best_params"], dtype=float)
        best_hc_value = train_payload["hill_climb"]["best_p11"]
        rb_value = train_payload["random_baseline"]["best_p11"]

        result = base_result("train")
        result.update(
            {
                "seed": args.seed,
                "max_evals": args.max_evals,
                "random_baseline_samples": args.random_baseline_samples,
                **train_payload,
                "best_reference": {
                    "best_overall": max(best_hc_value, rb_value),
                    "winner": (
                        "hill_climb" if best_hc_value >= rb_value else "random_baseline"
                    ),
                },
                "best_params": (
                    train_payload["hill_climb"]["best_params"]
                    if best_hc_value >= rb_value
                    else train_payload["random_baseline"]["best_params"]
                ),
                "best_metric": max(best_hc_value, rb_value),
                "strict_success_rate": 1.0 if max(best_hc_value, rb_value) >= METRIC_THRESHOLD else 0.0,
                "success_count": int(max(best_hc_value, rb_value) >= METRIC_THRESHOLD),
            }
        )

        save_circuit_if_requested(args.save_circuit, best_hc_params)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved train result: {output_path}")
        print(
            f"Hill-climb best P(|11>): {best_hc_value:.6f} | Random baseline best: {rb_value:.6f}"
        )
        print(f"Metric threshold: {METRIC_THRESHOLD:.2f}")
        return

    if args.mode == "proof":
        params = np.asarray(resolve_params_from_args(args), dtype=float)
        proof = run_proof_hardware(
            params=params,
            backend_name=args.backend,
            instance=args.instance,
            optimization_level=args.optimization_level,
            repetitions=args.proof_repetitions,
            save_circuit_path=args.save_circuit,
        )

        result = base_result("proof")
        result.update(
            {
                "seed": args.seed,
                "proof_repetitions": args.proof_repetitions,
                "backend": proof["backend"],
                "params": params.tolist(),
                "instance": proof["instance"],
                "job_count": proof["job_count"],
                "run_count": proof["run_count"],
                "job_id": proof["job_id_first"],
                "p11": proof["p11"],
                "abs_p11": proof["abs_p11_values"][0] if proof["run_count"] == 1 else mean(proof["abs_p11_values"]),
                "strict_success_rate": proof["strict_success_rate"],
                "success_count": proof["success_count"],
                "p11_std": proof["p11_std"],
                "p11_ci95": proof["p11_ci95"],
                "estimator_std_mean": proof["estimator_std_mean"],
                "transpiled_depth": proof["transpiled_depth"],
                "two_qubit_gate_count": proof["two_qubit_gate_count"],
                "run_summaries": proof["run_summaries"],
                "violates_metric": proof["violates_metric"],
                "all_abs_p11_values": proof["abs_p11_values"],
                "metric": "|11> probability",
            }
        )

        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved proof result: {output_path}")
        print(f"Backend: {proof['backend']}")
        print(f"Mean P(|11>): {proof['p11']:.6f} +/- {proof['p11_std']:.6f}")
        print(f"Proof strict success: {proof['success_count']}/{proof['run_count']}")
        print(f"Mean abs P(|11>): {mean(proof['abs_p11_values']):.6f}")
        print(f"Metric passed: {proof['violates_metric']}")
        return


if __name__ == "__main__":
    main()
