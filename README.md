# Colossus Training Flux

**Independent local training-capacity and gang-scheduling exhibit inspired by public distributed-training patterns.**

This repository is **not affiliated with, endorsed by, or operated by xAI**. It does not establish access to Colossus, xAI GPU clusters, NVLink/NVSwitch fabrics, NCCL telemetry, production schedulers, checkpoints, or proprietary infrastructure.

## Current capability

### Python capacity model

[`src/training_flux.py`](src/training_flux.py) deterministically models:

- priority ordering with deterministic tie-breaking;
- admission against an explicit MW power-budget assumption;
- a simple thermal-hold heuristic from ambient temperature and modeled admitted MW;
- fail-closed validation for invalid GPU counts, non-finite power/priority inputs, and invalid workloads.

Every result emits:

`MODELED_TRAINING_CAPACITY_SCENARIO_NOT_CLUSTER_RUNTIME`

The historical `schedule()` API remains as a compatibility alias for the same bounded model. It does not submit jobs to hardware.

### Go gang scheduler

[`src/gang_scheduler.go`](src/gang_scheduler.go) is a repository-local, in-memory scheduling mechanism. It now:

- validates nodes and jobs before mutation;
- rejects duplicate node/job identities;
- requires whole 8-GPU allocation chunks;
- chooses nodes deterministically;
- keeps each gang allocation inside one modeled cluster zone;
- preserves a pending job when same-zone capacity is insufficient;
- avoids nested read-lock acquisition when calculating statistics.

Its evidence token is:

`LOCAL_IN_MEMORY_GANG_SCHEDULER_NOT_CLUSTER_RUNTIME`

Fields such as `NVLinkSpeedGBps` and `MaxLatencyMs` are descriptive inputs only. The scheduler does not measure or enforce network behavior.

## Native proof

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
go test src/gang_scheduler.go src/gang_scheduler_test.go
```

The Public Truth Gate runs Python 3.11 and 3.13, exercises the Go scheduler tests, and binds pull-request proof to the exact source head and base ancestry.

## Evidence boundary

This repository does **not** claim:

- operation of a 100,000+ GPU cluster;
- xAI/Colossus deployment or affiliation;
- measured GPU utilization, throughput, latency, power, thermal, or training performance;
- NCCL, NVLink, NVSwitch, InfiniBand, RDMA, or network-topology measurement;
- distributed checkpoint management or checkpoint recovery;
- live elastic scaling or production preemption;
- MCP/APEX integration or autonomous external control;
- proprietary access, production credentials, or external deployment authority.

## Portfolio role

The transferable capability is **deterministic capacity admission plus same-zone in-memory gang scheduling with explicit proof boundaries**. Company naming provides problem context only; it does not establish a relationship with xAI.
