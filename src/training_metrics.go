/// Colossus Training Flux — Training Metrics & Monitoring
/// Collects, aggregates, and alerts on real training job metrics.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// TrainingMetric represents a single metric sample from a training job
type TrainingMetric struct {
	JobID       string            `json:"job_id"`
	Step        int64             `json:"step"`
	Timestamp   time.Time         `json:"timestamp"`
	Loss        float64           `json:"loss"`
	LearningRate float64          `json:"learning_rate"`
	Throughput  float64           `json:"throughput_tokens_per_sec"`
	GPUUtil     float64           `json:"gpu_utilization"`
	MemoryUsed  float64           `json:"memory_used_gb"`
	MemoryTotal float64           `json:"memory_total_gb"`
	GradientNorm float64          `json:"gradient_norm"`
	Custom      map[string]float64 `json:"custom,omitempty"`
}

// TrainingMetricsCollector aggregates metrics and detects anomalies
type TrainingMetricsCollector struct {
	jobID          string
	metrics        []TrainingMetric
	lossWindow     []float64
	windowSize     int
	alerts         []Alert
}

// Alert represents a detected anomaly or threshold breach
type Alert struct {
	Severity    string    `json:"severity"`    // "warning" or "critical"
	JobID       string    `json:"job_id"`
	Step        int64     `json:"step"`
	Rule        string    `json:"rule"`
	Message     string    `json:"message"`
	Timestamp   time.Time `json:"timestamp"`
}

// NewTrainingMetricsCollector creates a new collector for a training job
func NewTrainingMetricsCollector(jobID string) *TrainingMetricsCollector {
	return &TrainingMetricsCollector{
		jobID:      jobID,
		metrics:    make([]TrainingMetric, 0),
		lossWindow: make([]float64, 0),
		windowSize: 10,
		alerts:     make([]Alert, 0),
	}
}

// RecordMetric adds a metric sample and runs anomaly detection
func (c *TrainingMetricsCollector) RecordMetric(metric TrainingMetric) []Alert {
	c.metrics = append(c.metrics, metric)

	// Maintain rolling window for loss trend
	c.lossWindow = append(c.lossWindow, metric.Loss)
	if len(c.lossWindow) > c.windowSize {
		c.lossWindow = c.lossWindow[1:]
	}

	var newAlerts []Alert

	// 1. Loss explosion detection
	if len(c.lossWindow) >= 3 {
		avg := c.average(c.lossWindow[:len(c.lossWindow)-1])
		current := c.lossWindow[len(c.lossWindow)-1]
		if avg > 0 && current > avg*2.0 {
			alert := Alert{
				Severity:  "critical",
				JobID:     c.jobID,
				Step:      metric.Step,
				Rule:      "loss_explosion",
				Message:   fmt.Sprintf("Loss jumped from %.4f to %.4f (2x threshold)", avg, current),
				Timestamp: time.Now(),
			}
			c.alerts = append(c.alerts, alert)
			newAlerts = append(newAlerts, alert)
		}
	}

	// 2. NaN loss detection
	if math.IsNaN(metric.Loss) || math.IsInf(metric.Loss, 0) {
		alert := Alert{
			Severity:  "critical",
			JobID:     c.jobID,
			Step:      metric.Step,
			Rule:      "nan_loss",
			Message:   fmt.Sprintf("Loss is NaN/Inf at step %d", metric.Step),
			Timestamp: time.Now(),
		}
		c.alerts = append(c.alerts, alert)
		newAlerts = append(newAlerts, alert)
	}

	// 3. Gradient norm spike detection
	if metric.GradientNorm > 100.0 {
		alert := Alert{
			Severity:  "warning",
			JobID:     c.jobID,
			Step:      metric.Step,
			Rule:      "gradient_spike",
			Message:   fmt.Sprintf("Gradient norm %.2f exceeds 100.0", metric.GradientNorm),
			Timestamp: time.Now(),
		}
		c.alerts = append(c.alerts, alert)
		newAlerts = append(newAlerts, alert)
	}

	// 4. GPU utilization drop detection
	if metric.GPUUtil < 30.0 {
		alert := Alert{
			Severity:  "warning",
			JobID:     c.jobID,
			Step:      metric.Step,
			Rule:      "low_gpu_util",
			Message:   fmt.Sprintf("GPU utilization %.1f%% below 30%% threshold", metric.GPUUtil),
			Timestamp: time.Now(),
		}
		c.alerts = append(c.alerts, alert)
		newAlerts = append(newAlerts, alert)
	}

	// 5. Memory pressure detection
	if metric.MemoryTotal > 0 && metric.MemoryUsed/metric.MemoryTotal > 0.95 {
		alert := Alert{
			Severity:  "critical",
			JobID:     c.jobID,
			Step:      metric.Step,
			Rule:      "memory_pressure",
			Message:   fmt.Sprintf("Memory usage %.1f%% exceeds 95%%", metric.MemoryUsed/metric.MemoryTotal*100),
			Timestamp: time.Now(),
		}
		c.alerts = append(c.alerts, alert)
		newAlerts = append(newAlerts, alert)
	}

	return newAlerts
}

// GetSummary returns a summary of the training job metrics
func (c *TrainingMetricsCollector) GetSummary() map[string]interface{} {
	if len(c.metrics) == 0 {
		return map[string]interface{}{
			"job_id":     c.jobID,
			"total_steps": 0,
			"status":     "no_data",
		}
	}

	last := c.metrics[len(c.metrics)-1]
	first := c.metrics[0]

	avgLoss := c.averageFloat64(c.metrics, func(m TrainingMetric) float64 { return m.Loss })
	avgThroughput := c.averageFloat64(c.metrics, func(m TrainingMetric) float64 { return m.Throughput })
	avgGPU := c.averageFloat64(c.metrics, func(m TrainingMetric) float64 { return m.GPUUtil })

	summary := map[string]interface{}{
		"job_id":          c.jobID,
		"total_steps":     len(c.metrics),
		"first_step":      first.Step,
		"last_step":       last.Step,
		"final_loss":      last.Loss,
		"avg_loss":        avgLoss,
		"avg_throughput":  avgThroughput,
		"avg_gpu_util":    avgGPU,
		"final_lr":        last.LearningRate,
		"total_alerts":    len(c.alerts),
		"warning_alerts":  c.countAlertsBySeverity("warning"),
		"critical_alerts": c.countAlertsBySeverity("critical"),
		"status":          c.determineStatus(),
	}

	return summary
}

// GetAlerts returns all alerts, optionally filtered by severity
func (c *TrainingMetricsCollector) GetAlerts(severity string) []Alert {
	if severity == "" {
		return c.alerts
	}
	filtered := make([]Alert, 0)
	for _, a := range c.alerts {
		if a.Severity == severity {
			filtered = append(filtered, a)
		}
	}
	return filtered
}

// ExportJSON exports the metrics summary as JSON
func (c *TrainingMetricsCollector) ExportJSON() (string, error) {
	summary := c.GetSummary()
	summary["alerts"] = c.alerts
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Helper functions

func (c *TrainingMetricsCollector) average(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (c *TrainingMetricsCollector) averageFloat64(metrics []TrainingMetric, selector func(TrainingMetric) float64) float64 {
	if len(metrics) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, m := range metrics {
		sum += selector(m)
	}
	return sum / float64(len(metrics))
}

func (c *TrainingMetricsCollector) countAlertsBySeverity(severity string) int {
	count := 0
	for _, a := range c.alerts {
		if a.Severity == severity {
			count++
		}
	}
	return count
}

func (c *TrainingMetricsCollector) determineStatus() string {
	for _, a := range c.alerts {
		if a.Severity == "critical" {
			return "critical"
		}
	}
	if len(c.alerts) > 0 {
		return "warning"
	}
	return "healthy"
}

// --- Tests ---

func TestTrainingMetricsCollector_LossExplosion(t *testing.T) {
	collector := NewTrainingMetricsCollector("test-job-001")

	// Normal metrics
	for i := int64(0); i < 5; i++ {
		collector.RecordMetric(TrainingMetric{
			JobID:     "test-job-001",
			Step:      i,
			Timestamp: time.Now(),
			Loss:      1.0,
			GPUUtil:   80.0,
		})
	}

	// Loss explosion
	alerts := collector.RecordMetric(TrainingMetric{
		JobID:     "test-job-001",
		Step:      5,
		Timestamp: time.Now(),
		Loss:      3.0, // 3x the average
		GPUUtil:   80.0,
	})

	if len(alerts) == 0 {
		t.Fatal("Expected loss explosion alert")
	}
	if alerts[0].Rule != "loss_explosion" {
		t.Fatalf("Expected loss_explosion alert, got %s", alerts[0].Rule)
	}
}

func TestTrainingMetricsCollector_NaNDetection(t *testing.T) {
	collector := NewTrainingMetricsCollector("test-job-002")

	alerts := collector.RecordMetric(TrainingMetric{
		JobID:     "test-job-002",
		Step:      1,
		Timestamp: time.Now(),
		Loss:      math.NaN(),
		GPUUtil:   80.0,
	})

	if len(alerts) == 0 {
		t.Fatal("Expected NaN loss alert")
	}
	if alerts[0].Rule != "nan_loss" {
		t.Fatalf("Expected nan_loss alert, got %s", alerts[0].Rule)
	}
}

func TestTrainingMetricsCollector_Summary(t *testing.T) {
	collector := NewTrainingMetricsCollector("test-job-003")

	for i := int64(0); i < 10; i++ {
		collector.RecordMetric(TrainingMetric{
			JobID:       "test-job-003",
			Step:        i,
			Timestamp:   time.Now(),
			Loss:        2.0 - float64(i)*0.1,
			Throughput:  1000.0,
			GPUUtil:     85.0,
			MemoryUsed:  10.0,
			MemoryTotal: 16.0,
		})
	}

	summary := collector.GetSummary()
	if summary["total_steps"] != 10 {
		t.Fatalf("Expected 10 steps, got %v", summary["total_steps"])
	}
	if summary["status"] != "healthy" {
		t.Fatalf("Expected healthy status, got %v", summary["status"])
	}
}

func TestTrainingMetricsCollector_MemoryPressure(t *testing.T) {
	collector := NewTrainingMetricsCollector("test-job-004")

	alerts := collector.RecordMetric(TrainingMetric{
		JobID:       "test-job-004",
		Step:        1,
		Timestamp:   time.Now(),
		Loss:        1.0,
		GPUUtil:     80.0,
		MemoryUsed:  15.5,
		MemoryTotal: 16.0, // 96.9% - below threshold
	})
	if len(alerts) > 0 {
		t.Fatal("Expected no memory pressure alert at 96.9%")
	}

	alerts = collector.RecordMetric(TrainingMetric{
		JobID:       "test-job-004",
		Step:        2,
		Timestamp:   time.Now(),
		Loss:        1.0,
		GPUUtil:     80.0,
		MemoryUsed:  15.8,
		MemoryTotal: 16.0, // 98.75% - above threshold
	})
	if len(alerts) == 0 {
		t.Fatal("Expected memory pressure alert at 98.75%")
	}
	if alerts[0].Rule != "memory_pressure" {
		t.Fatalf("Expected memory_pressure alert, got %s", alerts[0].Rule)
	}
}
