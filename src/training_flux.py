#!/usr/bin/env python3
"""Training job flux scheduler under power + thermal caps — xAI/Colossus angle.

Packs jobs into a power budget with thermal headroom; first-principles caps.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

ANSWER = 42
CONFIDENCE_FLOOR = 0.31415
FLUX = 1.21
THROTTLE_C = 83.0

@dataclass
class TrainJob:
    name: str
    gpus: int
    watts_per_gpu: float
    priority: float
    est_hours: float

def schedule(jobs: list[TrainJob], max_mw: float, ambient_c: float = 28.0) -> dict:
    ordered = sorted(jobs, key=lambda j: -j.priority)
    used_w = 0.0
    plan = []
    for j in ordered:
        need = j.gpus * j.watts_per_gpu
        if used_w + need > max_mw * 1e6:
            plan.append({"job": j.name, "status": "QUEUED", "mw": 0})
            continue
        used_w += need
        # crude thermal proxy: more MW → higher outlet
        outlet = ambient_c + (used_w / 1e6) * 8.0
        status = "RUN" if outlet < THROTTLE_C else "THERMAL_HOLD"
        plan.append({"job": j.name, "status": status, "mw": round(need/1e6, 3), "outlet_c": round(outlet, 2)})
    util = used_w / (max_mw * 1e6)
    conf = max(CONFIDENCE_FLOOR, 1.0 - abs(util - 1/FLUX) * 0.5)
    return {"plan": plan, "util": round(util, 4), "confidence": round(conf, 4), "answer": ANSWER}

if __name__ == "__main__":
    jobs = [
        TrainJob("pretrain-a", 512, 700, 1.0, 48),
        TrainJob("sft-b", 128, 700, 0.7, 12),
        TrainJob("eval-c", 64, 500, 0.4, 4),
    ]
    print(schedule(jobs, max_mw=0.6))
