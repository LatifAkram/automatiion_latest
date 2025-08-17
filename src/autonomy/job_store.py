#!/usr/bin/env python3
"""
Autonomous Job Store (SQLite, stdlib only)
- Jobs: queued, running, completed, failed, cancelled
- Steps: per-job action execution trace with results
- Webhooks: per-job event notifications (created, completed, failed)
"""

import os
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

DB_PATH = Path("runs/autonomy.db")

SCHEMA_SQL = [
	"""
	CREATE TABLE IF NOT EXISTS jobs (
		id TEXT PRIMARY KEY,
		type TEXT NOT NULL,
		params TEXT NOT NULL,
		priority INTEGER DEFAULT 0,
		run_at REAL DEFAULT 0,
		status TEXT NOT NULL,
		created_at REAL NOT NULL,
		updated_at REAL NOT NULL,
		last_error TEXT
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS steps (
		job_id TEXT NOT NULL,
		step_index INTEGER NOT NULL,
		action TEXT NOT NULL,
		params TEXT NOT NULL,
		status TEXT NOT NULL,
		result TEXT,
		started_at REAL,
		finished_at REAL,
		PRIMARY KEY(job_id, step_index)
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS webhooks (
		job_id TEXT NOT NULL,
		url TEXT NOT NULL,
		event TEXT NOT NULL,
		PRIMARY KEY(job_id, url, event)
	);
	"""
]

@dataclass
class Job:
	id: str
	type: str
	params: Dict[str, Any]
	priority: int
	run_at: float
	status: str
	created_at: float
	updated_at: float
	last_error: Optional[str] = None

@dataclass
class Step:
	job_id: str
	step_index: int
	action: str
	params: Dict[str, Any]
	status: str
	result: Optional[Dict[str, Any]] = None
	started_at: Optional[float] = None
	finished_at: Optional[float] = None

class JobStore:
	def __init__(self, db_path: Path = DB_PATH):
		self.db_path = db_path
		self._lock = threading.Lock()
		self._ensure_db()

	def _ensure_db(self):
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		with sqlite3.connect(self.db_path) as conn:
			for stmt in SCHEMA_SQL:
				conn.execute(stmt)
			conn.commit()

	def create_job(self, job_id: str, job_type: str, params: Dict[str, Any], priority: int = 0, run_at: Optional[float] = None) -> Job:
		with self._lock, sqlite3.connect(self.db_path) as conn:
			created = time.time()
			run_at_val = float(run_at) if run_at is not None else created
			conn.execute(
				"INSERT INTO jobs (id, type, params, priority, run_at, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
				(job_id, job_type, json.dumps(params), int(priority), run_at_val, 'queued', created, created)
			)
			conn.commit()
			return Job(id=job_id, type=job_type, params=params, priority=priority, run_at=run_at_val, status='queued', created_at=created, updated_at=created)

	def add_webhook(self, job_id: str, url: str, event: str = 'completed') -> None:
		with self._lock, sqlite3.connect(self.db_path) as conn:
			conn.execute("INSERT OR IGNORE INTO webhooks (job_id, url, event) VALUES (?, ?, ?)", (job_id, url, event))
			conn.commit()

	def list_webhooks(self, job_id: str, event: str) -> List[str]:
		with sqlite3.connect(self.db_path) as conn:
			cur = conn.execute("SELECT url FROM webhooks WHERE job_id=? AND event=?", (job_id, event))
			return [row[0] for row in cur.fetchall()]

	def get_job(self, job_id: str) -> Optional[Job]:
		with sqlite3.connect(self.db_path) as conn:
			cur = conn.execute("SELECT id, type, params, priority, run_at, status, created_at, updated_at, last_error FROM jobs WHERE id=?", (job_id,))
			row = cur.fetchone()
			if not row:
				return None
			return Job(id=row[0], type=row[1], params=json.loads(row[2]), priority=row[3], run_at=row[4], status=row[5], created_at=row[6], updated_at=row[7], last_error=row[8])

	def update_job_status(self, job_id: str, status: str, last_error: Optional[str] = None) -> None:
		with self._lock, sqlite3.connect(self.db_path) as conn:
			conn.execute("UPDATE jobs SET status=?, updated_at=?, last_error=? WHERE id=?", (status, time.time(), last_error, job_id))
			conn.commit()

	def list_pending_jobs(self, limit: int = 10) -> List[Job]:
		with sqlite3.connect(self.db_path) as conn:
			cur = conn.execute("SELECT id, type, params, priority, run_at, status, created_at, updated_at, last_error FROM jobs WHERE status IN ('queued','retry') AND run_at <= ? ORDER BY priority DESC, created_at ASC LIMIT ?", (time.time(), limit))
			rows = cur.fetchall()
			return [Job(id=r[0], type=r[1], params=json.loads(r[2]), priority=r[3], run_at=r[4], status=r[5], created_at=r[6], updated_at=r[7], last_error=r[8]) for r in rows]

	def add_step(self, job_id: str, step_index: int, action: str, params: Dict[str, Any]) -> None:
		with self._lock, sqlite3.connect(self.db_path) as conn:
			conn.execute("INSERT OR REPLACE INTO steps (job_id, step_index, action, params, status, started_at) VALUES (?, ?, ?, ?, ?, ?)", (job_id, step_index, action, json.dumps(params), 'running', time.time()))
			conn.commit()

	def finish_step(self, job_id: str, step_index: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
		with self._lock, sqlite3.connect(self.db_path) as conn:
			conn.execute("UPDATE steps SET status=?, result=?, finished_at=? WHERE job_id=? AND step_index=?", (status, json.dumps(result or {}), time.time(), job_id, step_index))
			conn.commit()

	def list_steps(self, job_id: str) -> List[Step]:
		with sqlite3.connect(self.db_path) as conn:
			cur = conn.execute("SELECT job_id, step_index, action, params, status, result, started_at, finished_at FROM steps WHERE job_id=? ORDER BY step_index ASC", (job_id,))
			rows = cur.fetchall()
			result: List[Step] = []
			for r in rows:
				result.append(Step(job_id=r[0], step_index=r[1], action=r[2], params=json.loads(r[3]), status=r[4], result=json.loads(r[5]) if r[5] else None, started_at=r[6], finished_at=r[7]))
			return result