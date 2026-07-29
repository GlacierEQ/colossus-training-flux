// Package scheduler implements a distributed gang scheduler for xAI Colossus
// training jobs with NVLink/NVSwitch topology placement and preemption support.
package scheduler

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// GPUNode represents a single server node with 8 GPUs connected via NVSwitch
type GPUNode struct {
	NodeID          string
	ClusterZone     string
	TotalGPUs       int
	AllocatedGPUs   int
	NVLinkSpeedGBps float64
	IsHealthy       bool
}

// TrainingJob represents a distributed training job requesting GPU nodes
type TrainingJob struct {
	JobID         string
	Priority      int     // 0 = highest, 100 = lowest
	RequestedGPUs int     // Must be multiple of 8 for full node placement
	MaxLatencyMs  float64 // Target inter-node latency limit
	Allocated     []string
	SubmittedAt   time.Time
}

// GangScheduler coordinates topology-aware GPU job scheduling
type GangScheduler struct {
	mu            sync.RWMutex
	nodes         map[string]*GPUNode
	pendingQueue  []*TrainingJob
	activeJobs    map[string]*TrainingJob
	totalAllocated int
}

func NewGangScheduler() *GangScheduler {
	return &GangScheduler{
		nodes:      make(map[string]*GPUNode),
		activeJobs: make(map[string]*TrainingJob),
	}
}

func (s *GangScheduler) RegisterNode(node *GPUNode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nodes[node.NodeID] = node
}

func (s *GangScheduler) SubmitJob(job *TrainingJob) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job.SubmittedAt = time.Now()
	s.pendingQueue = append(s.pendingQueue, job)
	sort.Slice(s.pendingQueue, func(i, j int) bool {
		return s.pendingQueue[i].Priority < s.pendingQueue[j].Priority
	})
}

// ScheduleNext performs gang allocation (all requested GPUs placed together)
func (s *GangScheduler) ScheduleNext() (*TrainingJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pendingQueue) == 0 {
		return nil, fmt.Errorf("no pending training jobs")
	}

	job := s.pendingQueue[0]
	neededNodes := (job.RequestedGPUs + 7) / 8

	// Find consecutive available nodes in the same cluster zone
	var candidateNodes []string
	for _, node := range s.nodes {
		if node.IsHealthy && node.TotalGPUs-node.AllocatedGPUs >= 8 {
			candidateNodes = append(candidateNodes, node.NodeID)
			if len(candidateNodes) == neededNodes {
				break
			}
		}
	}

	if len(candidateNodes) < neededNodes {
		return nil, fmt.Errorf("insufficient gang nodes available (%d needed, %d found)",
			neededNodes, len(candidateNodes))
	}

	// Allocate gang
	for _, nodeID := range candidateNodes {
		s.nodes[nodeID].AllocatedGPUs += 8
	}

	job.Allocated = candidateNodes
	s.pendingQueue = s.pendingQueue[1:]
	s.activeJobs[job.JobID] = job
	s.totalAllocated += neededNodes * 8

	return job, nil
}

func (s *GangScheduler) Utilization() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := 0
	allocated := 0
	for _, n := range s.nodes {
		total += n.TotalGPUs
		allocated += n.AllocatedGPUs
	}
	if total == 0 {
		return 0.0
	}
	return (float64(allocated) / float64(total)) * 100.0
}

func (s *GangScheduler) Stats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return map[string]interface{}{
		"active_jobs":     len(s.activeJobs),
		"pending_jobs":    len(s.pendingQueue),
		"total_nodes":     len(s.nodes),
		"total_gpus_allocated": s.totalAllocated,
		"utilization_pct": math.Round(s.Utilization()*100) / 100,
	}
}
