#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
python -m unittest discover -s tests -p 'test_*.py' -v
unformatted="$(gofmt -l src/gang_scheduler.go src/gang_scheduler_test.go)"
test -z "$unformatted"
go test src/gang_scheduler.go src/gang_scheduler_test.go

grep -q 'MODELED_TRAINING_CAPACITY_SCENARIO_NOT_CLUSTER_RUNTIME' README.md
grep -q 'MODELED_TRAINING_CAPACITY_SCENARIO_NOT_CLUSTER_RUNTIME' src/training_flux.py
grep -q 'LOCAL_IN_MEMORY_GANG_SCHEDULER_NOT_CLUSTER_RUNTIME' README.md
grep -q 'not affiliated with, endorsed by, or operated by xAI' README.md
! grep -q 'answer' src/training_flux.py
