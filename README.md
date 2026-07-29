# Colossus Training Flux — Distributed Training Job Scheduler ⚡

> **Priority-aware distributed training job orchestration across 100,000+ GPU clusters.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8)]()
[![Domain](https://img.shields.io/badge/Domain-Distributed%20Training-red)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements a **distributed training job scheduler** — the orchestration layer that assigns GPU resources, manages checkpointing, and coordinates data-parallel training across massive clusters. It demonstrates:

- **Multi-tenant job scheduling** with priority queues and preemption policies
- **GPU topology-aware placement** optimizing for NVLink/NVSwitch locality
- **Checkpoint management** with async saving and automatic recovery from failures
- **Elastic scaling** dynamically adjusting worker count based on cluster pressure

**Why this matters**: Training LLMs at scale is one of the most complex scheduling problems in computing. This codebase demonstrates the same **resource management, fault tolerance, and distributed coordination** skills needed for cloud infrastructure, HPC, and data pipeline engineering.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/training_flux.py` | Python | Job scheduler, checkpoint management, elastic scaling |
| `src/gpu_scheduler.go` | Go | Topology-aware GPU placement with concurrent job tracking |
| `tests/` | Python | Multi-job scheduling simulation with fault injection |

### Key Design

- **Gang scheduling**: All GPUs for a job start simultaneously — no straggler waste
- **NCCL-aware placement**: Jobs placed to minimize inter-node communication hops
- **Checkpoint sharding**: Model state distributed across workers for parallel save/load

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `job_status(job_id)` — training progress queryable by orchestrator agents
- **Mastermind Sidecar**: Publishes training metrics to APEX Highway mesh
- **AI Extension**: RL-based scheduling policy that learns optimal GPU placement from historical job traces

```python
status = await mcp_client.call_tool("training-flux", "cluster_utilization")
# Returns: {"gpu_utilization": 0.94, "active_jobs": 127, "queued": 23, "preempted": 2}
```

---

## ⚡ Quick Start

```bash
python3 src/training_flux.py
python3 tests/test_training_flux.py
```
