#!/usr/bin/env python3
"""Deterministic training-capacity scenario model.

This module does not operate an xAI/Colossus cluster, execute distributed
training, measure GPU telemetry, coordinate NCCL/NVLink, save checkpoints, or
perform elastic scaling. It models priority ordering, a power budget, and a
simple thermal hold heuristic from explicit local assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

<<<<<<< HEAD
EVIDENCE_STATE = "MODELED_TRAINING_CAPACITY_SCENARIO_NOT_CLUSTER_RUNTIME"
THERMAL_HOLD_C = 83.0
=======
CONFIDENCE_FLOOR = 0.31415
FLUX = 1.21
THROTTLE_C = 83.0
>>>>>>> 13c9a77 (chore: Hyper Excellence Activation & structural matrix alignment)


@dataclass(frozen=True)
class TrainJob:
    name: str
    gpus: int
    watts_per_gpu: float
    priority: float
    est_hours: float


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _positive_finite(name: str, value: float) -> float:
    numeric = _finite(name, value)
    if numeric <= 0:
        raise ValueError(f"{name} must be > 0")
    return numeric


def _validate_job(job: TrainJob) -> None:
    if not job.name.strip():
        raise ValueError("job name must not be empty")
    if isinstance(job.gpus, bool) or not isinstance(job.gpus, int) or job.gpus < 1:
        raise ValueError("job gpus must be a positive integer")
    _positive_finite("watts_per_gpu", job.watts_per_gpu)
    _finite("priority", job.priority)
    _positive_finite("est_hours", job.est_hours)


def model_capacity_schedule(
    jobs: list[TrainJob], max_mw: float, ambient_c: float = 28.0
) -> dict:
    """Model admission under explicit power and thermal assumptions."""

    max_mw = _positive_finite("max_mw", max_mw)
    ambient_c = _finite("ambient_c", ambient_c)
    if not jobs:
        raise ValueError("at least one job is required")
    for job in jobs:
        _validate_job(job)

    ordered = sorted(jobs, key=lambda job: (-job.priority, job.name))
    max_watts = max_mw * 1_000_000.0
    used_watts = 0.0
    plan: list[dict] = []

    for job in ordered:
        need_watts = job.gpus * float(job.watts_per_gpu)
        prospective_watts = used_watts + need_watts
        if prospective_watts > max_watts:
            plan.append(
                {
                    "job": job.name,
                    "decision": "POWER_BUDGET_HOLD",
                    "modeled_mw": 0.0,
                }
            )
            continue
<<<<<<< HEAD

        modeled_outlet_c = ambient_c + (prospective_watts / 1_000_000.0) * 8.0
        if modeled_outlet_c >= THERMAL_HOLD_C:
            plan.append(
                {
                    "job": job.name,
                    "decision": "THERMAL_MODEL_HOLD",
                    "modeled_mw": 0.0,
                    "modeled_outlet_c": round(modeled_outlet_c, 2),
                }
            )
            continue

        used_watts = prospective_watts
        plan.append(
            {
                "job": job.name,
                "decision": "MODELED_ADMIT",
                "modeled_mw": round(need_watts / 1_000_000.0, 6),
                "modeled_outlet_c": round(modeled_outlet_c, 2),
            }
        )

    return {
        "plan": plan,
        "modeled_used_mw": round(used_watts / 1_000_000.0, 6),
        "modeled_utilization": round(used_watts / max_watts, 6),
        "power_budget_mw": max_mw,
        "ambient_assumption_c": ambient_c,
        "thermal_hold_assumption_c": THERMAL_HOLD_C,
        "evidence_state": EVIDENCE_STATE,
    }


def schedule(jobs: list[TrainJob], max_mw: float, ambient_c: float = 28.0) -> dict:
    """Compatibility alias for the historical public API."""

    return model_capacity_schedule(jobs, max_mw=max_mw, ambient_c=ambient_c)

=======
        used_w += need
        # crude thermal proxy: more MW → higher outlet
        outlet = ambient_c + (used_w / 1e6) * 8.0
        status = "RUN" if outlet < THROTTLE_C else "THERMAL_HOLD"
        plan.append({"job": j.name, "status": status, "mw": round(need/1e6, 3), "outlet_c": round(outlet, 2)})
    util = used_w / (max_mw * 1e6)
    conf = max(CONFIDENCE_FLOOR, 1.0 - abs(util - 1/FLUX) * 0.5)
    return {"plan": plan, "util": round(util, 4), "confidence": round(conf, 4) }
>>>>>>> 13c9a77 (chore: Hyper Excellence Activation & structural matrix alignment)

if __name__ == "__main__":
    demo_jobs = [
        TrainJob("pretrain-a", 512, 700, 1.0, 48),
        TrainJob("sft-b", 128, 700, 0.7, 12),
        TrainJob("eval-c", 64, 500, 0.4, 4),
    ]
    print(model_capacity_schedule(demo_jobs, max_mw=0.6))
