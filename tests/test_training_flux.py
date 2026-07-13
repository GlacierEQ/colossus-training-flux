import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from training_flux import TrainJob, schedule, ANSWER

def test_budget():
    jobs = [TrainJob("a", 1000, 700, 1.0, 1), TrainJob("b", 1000, 700, 0.5, 1)]
    r = schedule(jobs, max_mw=0.5)
    assert r["answer"]==ANSWER
    assert any(p["status"]=="QUEUED" for p in r["plan"]) or r["util"] <= 1.0

if __name__=="__main__":
    test_budget(); print("ok")
