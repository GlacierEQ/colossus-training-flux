package scheduler

import "testing"

func TestSameZoneAllocationAndDeterministicNodeOrder(t *testing.T) {
	s := NewGangScheduler()
	for _, node := range []*GPUNode{
		{NodeID: "b2", ClusterZone: "b", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "a2", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "b1", ClusterZone: "b", TotalGPUs: 8, IsHealthy: true},
	} {
		if err := s.RegisterNode(node); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.SubmitJob(&TrainingJob{JobID: "job", Priority: 1, RequestedGPUs: 16}); err != nil {
		t.Fatal(err)
	}
	job, err := s.ScheduleNext()
	if err != nil {
		t.Fatal(err)
	}
	if len(job.Allocated) != 2 || job.Allocated[0] != "a1" || job.Allocated[1] != "a2" {
		t.Fatalf("unexpected allocation: %#v", job.Allocated)
	}
}

func TestHigherPrioritySchedulesFirstWithStableTieBreak(t *testing.T) {
	s := NewGangScheduler()
	for _, node := range []*GPUNode{
		{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "a2", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "a3", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
	} {
		if err := s.RegisterNode(node); err != nil {
			t.Fatal(err)
		}
	}
	for _, job := range []*TrainingJob{
		{JobID: "low", Priority: 10, RequestedGPUs: 8},
		{JobID: "z-high", Priority: 90, RequestedGPUs: 8},
		{JobID: "a-high", Priority: 90, RequestedGPUs: 8},
	} {
		if err := s.SubmitJob(job); err != nil {
			t.Fatal(err)
		}
	}
	first, err := s.ScheduleNext()
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.ScheduleNext()
	if err != nil {
		t.Fatal(err)
	}
	if first.JobID != "a-high" || second.JobID != "z-high" {
		t.Fatalf("unexpected priority order: %s then %s", first.JobID, second.JobID)
	}
}

func TestRegisteredNodeAndSubmittedJobAreCopied(t *testing.T) {
	s := NewGangScheduler()
	node := &GPUNode{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true}
	if err := s.RegisterNode(node); err != nil {
		t.Fatal(err)
	}
	node.ClusterZone = "mutated"
	node.IsHealthy = false
	node.TotalGPUs = 800

	job := &TrainingJob{JobID: "job", Priority: 50, RequestedGPUs: 8}
	if err := s.SubmitJob(job); err != nil {
		t.Fatal(err)
	}
	job.JobID = "mutated"
	job.Priority = 0
	job.RequestedGPUs = 800

	allocated, err := s.ScheduleNext()
	if err != nil {
		t.Fatal(err)
	}
	if allocated.JobID != "job" || allocated.RequestedGPUs != 8 {
		t.Fatalf("caller mutation leaked into scheduler: %#v", allocated)
	}
	if len(allocated.Allocated) != 1 || allocated.Allocated[0] != "a1" {
		t.Fatalf("registered node mutation leaked into scheduler: %#v", allocated.Allocated)
	}

	allocated.Allocated[0] = "caller-mutated"
	stats := s.Stats()
	if stats["active_jobs"].(int) != 1 || stats["total_gpus_allocated"].(int) != 8 {
		t.Fatalf("returned job mutation affected internal state: %#v", stats)
	}
}

func TestInsufficientSameZoneCapacityPreservesPendingJob(t *testing.T) {
	s := NewGangScheduler()
	for _, node := range []*GPUNode{
		{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true},
		{NodeID: "b1", ClusterZone: "b", TotalGPUs: 8, IsHealthy: true},
	} {
		if err := s.RegisterNode(node); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.SubmitJob(&TrainingJob{JobID: "job", Priority: 1, RequestedGPUs: 16}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ScheduleNext(); err == nil {
		t.Fatal("expected same-zone capacity failure")
	}
	stats := s.Stats()
	if stats["pending_jobs"].(int) != 1 || stats["active_jobs"].(int) != 0 {
		t.Fatalf("job was not preserved: %#v", stats)
	}
}

func TestValidationAndDuplicateProtection(t *testing.T) {
	s := NewGangScheduler()
	if err := s.RegisterNode(nil); err == nil {
		t.Fatal("nil node should fail")
	}
	if err := s.RegisterNode(&GPUNode{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true}); err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterNode(&GPUNode{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true}); err == nil {
		t.Fatal("duplicate node should fail")
	}
	if err := s.SubmitJob(nil); err == nil {
		t.Fatal("nil job should fail")
	}
	if err := s.SubmitJob(&TrainingJob{JobID: "bad", Priority: 1, RequestedGPUs: 7}); err == nil {
		t.Fatal("non-node-aligned GPU request should fail")
	}
	job := &TrainingJob{JobID: "job", Priority: 1, RequestedGPUs: 8}
	if err := s.SubmitJob(job); err != nil {
		t.Fatal(err)
	}
	if err := s.SubmitJob(&TrainingJob{JobID: "job", Priority: 2, RequestedGPUs: 8}); err == nil {
		t.Fatal("duplicate pending job should fail")
	}
}

func TestStatsDoNotDeadlockAndExposeLocalEvidenceState(t *testing.T) {
	s := NewGangScheduler()
	if err := s.RegisterNode(&GPUNode{NodeID: "a1", ClusterZone: "a", TotalGPUs: 8, IsHealthy: true}); err != nil {
		t.Fatal(err)
	}
	if err := s.SubmitJob(&TrainingJob{JobID: "job", Priority: 1, RequestedGPUs: 8}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ScheduleNext(); err != nil {
		t.Fatal(err)
	}
	stats := s.Stats()
	if stats["utilization_pct"].(float64) != 100.0 {
		t.Fatalf("unexpected utilization: %#v", stats)
	}
	if stats["evidence_state"] != "LOCAL_IN_MEMORY_GANG_SCHEDULER_NOT_CLUSTER_RUNTIME" {
		t.Fatalf("unexpected evidence state: %#v", stats)
	}
}
