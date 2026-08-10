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
