"""
Database Manager
===============

SQLite database management for workflows, execution logs, and performance metrics.
"""

import asyncio
import logging
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from ..models.workflow import Workflow, WorkflowStatus
from ..models.execution import ExecutionResult, PerformanceMetrics


class DatabaseManager:
    """Manages SQLite database operations for the automation platform."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = config.sqlite_path
        self.connection: Optional[sqlite3.Connection] = None
        
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise
            
    async def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Workflows table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                domain TEXT DEFAULT 'general',
                status TEXT DEFAULT 'planning',
                parameters TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                created_by TEXT,
                tags TEXT
            )
        """)
        
        # Execution results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_results (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                steps TEXT,
                errors TEXT,
                warnings TEXT,
                duration REAL DEFAULT 0.0,
                start_time TEXT,
                end_time TEXT,
                artifacts TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                key TEXT PRIMARY KEY,
                metrics TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,
                hash TEXT
            )
        """)
        
        self.connection.commit()
        
    async def save_workflow(self, workflow: Workflow):
        """Save workflow to database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO workflows 
                (id, name, description, domain, status, parameters, created_at, updated_at, started_at, completed_at, created_by, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.id,
                workflow.name,
                workflow.description,
                workflow.domain,
                workflow.status.value,
                str(workflow.parameters),
                workflow.created_at.isoformat(),
                workflow.updated_at.isoformat(),
                workflow.started_at.isoformat() if workflow.started_at else None,
                workflow.completed_at.isoformat() if workflow.completed_at else None,
                workflow.created_by,
                str(workflow.tags)
            ))
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow: {e}", exc_info=True)
            raise
            
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_workflow(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow: {e}", exc_info=True)
            raise
            
    async def get_active_workflows(self) -> List[Workflow]:
        """Get all active workflows."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM workflows 
                WHERE status IN ('planning', 'executing', 'paused')
                ORDER BY created_at DESC
            """)
            
            workflows = []
            for row in cursor.fetchall():
                workflows.append(self._row_to_workflow(row))
                
            return workflows
            
        except Exception as e:
            self.logger.error(f"Failed to get active workflows: {e}", exc_info=True)
            raise
            
    async def update_workflow_status(self, workflow_id: str, status: WorkflowStatus):
        """Update workflow status."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE workflows 
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status.value, datetime.utcnow().isoformat(), workflow_id))
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update workflow status: {e}", exc_info=True)
            raise
            
    async def save_execution_result(self, result: ExecutionResult):
        """Save execution result to database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO execution_results 
                (id, workflow_id, success, steps, errors, warnings, duration, start_time, end_time, artifacts, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"exec_{result.workflow_id}_{datetime.utcnow().timestamp()}",
                result.workflow_id,
                result.success,
                str(result.steps),
                str(result.errors),
                str(result.warnings),
                result.duration,
                result.start_time.isoformat() if result.start_time else None,
                result.end_time.isoformat() if result.end_time else None,
                str(result.artifacts),
                str(result.metadata),
                result.created_at.isoformat()
            ))
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save execution result: {e}", exc_info=True)
            raise
            
    async def get_latest_execution_result(self, workflow_id: str) -> Optional[ExecutionResult]:
        """Get latest execution result for a workflow."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM execution_results 
                WHERE workflow_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (workflow_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_execution_result(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get execution result: {e}", exc_info=True)
            raise
            
    async def save_performance_metrics(self, key: str, metrics: Dict[str, Any]):
        """Save performance metrics."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_metrics 
                (key, metrics, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                key,
                str(metrics),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            ))
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}", exc_info=True)
            raise
            
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT key, metrics FROM performance_metrics")
            
            metrics = {}
            for row in cursor.fetchall():
                # Parse metrics string back to dict
                import ast
                metrics[row['key']] = ast.literal_eval(row['metrics'])
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
            return {}
            
    async def save_workflow_plan(self, workflow_id: str, plan: Dict[str, Any]):
        """Save workflow execution plan."""
        # This would typically store the plan in a separate table
        # For now, just log it
        self.logger.info(f"Saved plan for workflow {workflow_id}")
        
    def _row_to_workflow(self, row) -> Workflow:
        """Convert database row to Workflow object."""
        import ast
        
        return Workflow(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            domain=row['domain'],
            status=WorkflowStatus(row['status']),
            parameters=ast.literal_eval(row['parameters']) if row['parameters'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            created_by=row['created_by'],
            tags=ast.literal_eval(row['tags']) if row['tags'] else []
        )
        
    def _row_to_execution_result(self, row) -> ExecutionResult:
        """Convert database row to ExecutionResult object."""
        import ast
        
        return ExecutionResult(
            workflow_id=row['workflow_id'],
            success=bool(row['success']),
            steps=ast.literal_eval(row['steps']) if row['steps'] else [],
            errors=ast.literal_eval(row['errors']) if row['errors'] else [],
            warnings=ast.literal_eval(row['warnings']) if row['warnings'] else [],
            duration=float(row['duration']),
            start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            artifacts=ast.literal_eval(row['artifacts']) if row['artifacts'] else {},
            metadata=ast.literal_eval(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at'])
        )
        
    async def shutdown(self):
        """Shutdown database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")