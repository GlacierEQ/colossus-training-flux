from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
<<<<<<< HEAD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
=======
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from training_flux import TrainJob, schedule

def test_budget():
    jobs = [TrainJob("a", 1000, 700, 1.0, 1), TrainJob("b", 1000, 700, 0.5, 1)]
    r = schedule(jobs, max_mw=0.5)
    assert any(p["status"]=="QUEUED" for p in r["plan"]) or r["util"] <= 1.0
>>>>>>> 13c9a77 (chore: Hyper Excellence Activation & structural matrix alignment)

from training_flux import TrainJob, model_capacity_schedule, schedule


class TestTrainingCapacityModel(unittest.TestCase):
    def test_power_budget_holds_excess_jobs(self) -> None:
        jobs = [
            TrainJob("a", 1000, 700, 1.0, 1),
            TrainJob("b", 1000, 700, 0.5, 1),
        ]
        result = model_capacity_schedule(jobs, max_mw=0.5)
        self.assertEqual(
            result["evidence_state"],
            "MODELED_TRAINING_CAPACITY_SCENARIO_NOT_CLUSTER_RUNTIME",
        )
        self.assertLessEqual(result["modeled_utilization"], 1.0)
        self.assertTrue(
            any(item["decision"] == "POWER_BUDGET_HOLD" for item in result["plan"])
        )
        self.assertNotIn("answer", result)
        self.assertNotIn("confidence", result)

    def test_thermal_hold_does_not_consume_modeled_budget(self) -> None:
        jobs = [TrainJob("hot", 1000, 700, 1.0, 1)]
        result = model_capacity_schedule(jobs, max_mw=1.0, ambient_c=80.0)
        self.assertEqual(result["plan"][0]["decision"], "THERMAL_MODEL_HOLD")
        self.assertEqual(result["modeled_used_mw"], 0.0)
        self.assertEqual(result["modeled_utilization"], 0.0)

    def test_priority_order_is_deterministic(self) -> None:
        jobs = [
            TrainJob("z", 1, 100, 1.0, 1),
            TrainJob("a", 1, 100, 1.0, 1),
        ]
        result = model_capacity_schedule(jobs, max_mw=1.0)
        self.assertEqual([item["job"] for item in result["plan"]], ["a", "z"])

    def test_historical_schedule_api_is_bounded_alias(self) -> None:
        jobs = [TrainJob("a", 1, 100, 1.0, 1)]
        self.assertEqual(
            schedule(jobs, max_mw=1.0),
            model_capacity_schedule(jobs, max_mw=1.0),
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        valid = [TrainJob("a", 1, 100, 1.0, 1)]
        for max_mw in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                model_capacity_schedule(valid, max_mw=max_mw)
        with self.assertRaises(ValueError):
            model_capacity_schedule([], max_mw=1.0)
        with self.assertRaises(ValueError):
            model_capacity_schedule([TrainJob("", 1, 100, 1.0, 1)], max_mw=1.0)
        with self.assertRaises(ValueError):
            model_capacity_schedule([TrainJob("a", 0, 100, 1.0, 1)], max_mw=1.0)
        with self.assertRaises(ValueError):
            model_capacity_schedule([TrainJob("a", 1, math.nan, 1.0, 1)], max_mw=1.0)
        with self.assertRaises(ValueError):
            model_capacity_schedule([TrainJob("a", 1, 100, math.inf, 1)], max_mw=1.0)


if __name__ == "__main__":
    unittest.main()
