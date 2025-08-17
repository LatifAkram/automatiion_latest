#!/usr/bin/env python3
"""
Benchmark Metrics Recorder
- Captures per-step results: action, success, timings, healing usage
- Captures system metrics before/after (from builtin performance monitor)
- Produces JSON-serializable report with summary stats (p50/p95 latency, success rate, healing rate)
"""

import time
import statistics
from typing import Any, Dict, List, Optional

try:
    from core.builtin_performance_monitor import get_system_metrics_dict
except Exception:
    def get_system_metrics_dict() -> Dict[str, Any]:
        return {}

class BenchmarkRecorder:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.pre_metrics: Dict[str, Any] = {}
        self.post_metrics: Dict[str, Any] = {}
        self.steps: List[Dict[str, Any]] = []

    def start(self):
        self.start_time = time.time()
        self.pre_metrics = get_system_metrics_dict()

    def record_step(self, action: str, success: bool, execution_time_ms: float, healing_used: bool = False, extra: Optional[Dict[str, Any]] = None):
        self.steps.append({
            'action': action,
            'success': bool(success),
            'execution_time_ms': float(execution_time_ms),
            'healing_used': bool(healing_used),
            'extra': extra or {}
        })

    def finish(self) -> Dict[str, Any]:
        self.end_time = time.time()
        self.post_metrics = get_system_metrics_dict()
        durations = [s['execution_time_ms'] for s in self.steps if isinstance(s.get('execution_time_ms'), (int, float))]
        successes = sum(1 for s in self.steps if s['success'])
        total = len(self.steps) or 1
        healing_count = sum(1 for s in self.steps if s.get('healing_used'))
        summary = {
            'name': self.name,
            'start_ts': self.start_time,
            'end_ts': self.end_time,
            'duration_s': (self.end_time - self.start_time) if self.end_time and self.start_time else None,
            'steps_total': len(self.steps),
            'steps_success': successes,
            'success_rate': successes / total * 100.0,
            'healing_used_count': healing_count,
            'healing_rate': healing_count / total * 100.0 if total else 0.0,
            'latency_ms': {
                'p50': statistics.median(durations) if durations else 0.0,
                'p95': statistics.quantiles(durations, n=100)[94] if len(durations) >= 100 else (max(durations) if durations else 0.0),
                'avg': statistics.mean(durations) if durations else 0.0
            }
        }
        return {
            'summary': summary,
            'steps': self.steps,
            'system_metrics': {
                'before': self.pre_metrics,
                'after': self.post_metrics
            }
        }