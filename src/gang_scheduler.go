// Package scheduler provides a repository-local in-memory gang-scheduling model.
//
// It does not operate xAI Colossus, communicate over NVLink/NVSwitch, measure
// inter-node latency, or submit jobs to a production cluster.
package scheduler

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// GPUNode is a modeled scheduling node. Network fields are descriptive inputs
// only; this scheduler does not measure or control network hardware.
type GPUNode struct {
	NodeID          string
	ClusterZone     string
	TotalGPUs       int
	AllocatedGPUs   int
	NVLinkSpeedGBps float64
	IsHealthy       bool
}

// TrainingJob is a local scheduling request.
type TrainingJob struct {
	JobID         string
	Priority      int
	RequestedGPUs int
	MaxLatencyMs  float64
	Allocated     []string
	SubmittedAt   time.Time
}

// GangScheduler coordinates deterministic same-zone, full-node allocation.
type GangScheduler struct {
	mu             sync.RWMutex
	nodes          map[string]*GPUNode
	pendingQueue   []*TrainingJob
	activeJobs     map[string]*TrainingJob
	totalAllocated int
}

func NewGangScheduler() *GangScheduler {
	return &GangScheduler{
		nodes:      make(map[string]*GPUNode),
		activeJobs: make(map[string]*TrainingJob),
	}
}

func (s *GangScheduler) RegisterNode(node *GPUNode) error {
	if node == nil {
		return fmt.Errorf("node must not be nil")
	}
	if strings.TrimSpace(node.NodeID) == "" || strings.TrimSpace(node.ClusterZone) == "" {
		return fmt.Errorf("node ID and cluster zone are required")
	}
	if node.TotalGPUs < 8 || node.TotalGPUs%8 != 0 {
		return fmt.Errorf("node total GPUs must be a positive multiple of 8")
	}
	if node.AllocatedGPUs < 0 || node.AllocatedGPUs > node.TotalGPUs || node.AllocatedGPUs%8 != 0 {
		return fmt.Errorf("allocated GPUs must be a valid multiple of 8")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.nodes[node.NodeID]; exists {
		return fmt.Errorf("duplicate node ID: %s", node.NodeID)
	}
	copyNode := *node
	s.nodes[copyNode.NodeID] = &copyNode
	return nil
}

func (s *GangScheduler) SubmitJob(job *TrainingJob) error {
	if job == nil {
		return fmt.Errorf("job must not be nil")
	}
	if strings.TrimSpace(job.JobID) == "" {
		return fmt.Errorf("job ID is required")
	}
	if job.Priority < 0 || job.Priority > 100 {
		return fmt.Errorf("priority must be between 0 and 100")
	}
	if job.RequestedGPUs < 8 || job.RequestedGPUs%8 != 0 {
		return fmt.Errorf("requested GPUs must be a positive multiple of 8")
	}
	if job.MaxLatencyMs < 0 {
		return fmt.Errorf("max latency must not be negative")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.activeJobs[job.JobID]; exists {
		return fmt.Errorf("duplicate active job ID: %s", job.JobID)
	}
	for _, pending := range s.pendingQueue {
		if pending.JobID == job.JobID {
			return fmt.Errorf("duplicate pending job ID: %s", job.JobID)
		}
	}

	copyJob := *job
	copyJob.Allocated = append([]string(nil), job.Allocated...)
	copyJob.SubmittedAt = time.Now()
	s.pendingQueue = append(s.pendingQueue, &copyJob)
	sort.SliceStable(s.pendingQueue, func(i, j int) bool {
		if s.pendingQueue[i].Priority == s.pendingQueue[j].Priority {
			return s.pendingQueue[i].JobID < s.pendingQueue[j].JobID
		}
		return s.pendingQueue[i].Priority > s.pendingQueue[j].Priority
	})
	return nil
}

// ScheduleNext allocates all requested GPUs from full 8-GPU chunks in one zone.
func (s *GangScheduler) ScheduleNext() (*TrainingJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pendingQueue) == 0 {
		return nil, fmt.Errorf("no pending training jobs")
	}

	job := s.pendingQueue[0]
	neededNodes := job.RequestedGPUs / 8

	zoneNodes := make(map[string][]string)
	for nodeID, node := range s.nodes {
		if node.IsHealthy && node.TotalGPUs-node.AllocatedGPUs >= 8 {
			zoneNodes[node.ClusterZone] = append(zoneNodes[node.ClusterZone], nodeID)
		}
	}

	zones := make([]string, 0, len(zoneNodes))
	for zone := range zoneNodes {
		zones = append(zones, zone)
	}
	sort.Strings(zones)

	var candidateNodes []string
	for _, zone := range zones {
		nodeIDs := zoneNodes[zone]
		sort.Strings(nodeIDs)
		if len(nodeIDs) >= neededNodes {
			candidateNodes = append(candidateNodes, nodeIDs[:neededNodes]...)
			break
		}
	}

	if len(candidateNodes) < neededNodes {
		return nil, fmt.Errorf(
			"insufficient same-zone gang nodes available (%d needed)", neededNodes,
		)
	}

	for _, nodeID := range candidateNodes {
		s.nodes[nodeID].AllocatedGPUs += 8
	}

	job.Allocated = append([]string(nil), candidateNodes...)
	s.pendingQueue = s.pendingQueue[1:]
	s.activeJobs[job.JobID] = job
	s.totalAllocated += job.RequestedGPUs

	result := *job
	result.Allocated = append([]string(nil), job.Allocated...)
	return &result, nil
}

func (s *GangScheduler) utilizationLocked() float64 {
	total := 0
	allocated := 0
	for _, node := range s.nodes {
		total += node.TotalGPUs
		allocated += node.AllocatedGPUs
	}
	if total == 0 {
		return 0.0
	}
	return (float64(allocated) / float64(total)) * 100.0
}

func (s *GangScheduler) Utilization() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.utilizationLocked()
}

func (s *GangScheduler) Stats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return map[string]interface{}{
		"active_jobs":          len(s.activeJobs),
		"pending_jobs":         len(s.pendingQueue),
		"total_nodes":          len(s.nodes),
		"total_gpus_allocated": s.totalAllocated,
		"utilization_pct":      s.utilizationLocked(),
		"evidence_state":       "LOCAL_IN_MEMORY_GANG_SCHEDULER_NOT_CLUSTER_RUNTIME",
	}
}
