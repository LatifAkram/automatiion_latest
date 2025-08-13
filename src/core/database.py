"""
Database Manager
===============

SQLite database manager for storing workflows, execution logs, and performance metrics.
"""

import asyncio
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Use absolute imports to fix the relative import issue
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.workflow import Workflow, WorkflowStatus
from src.models.task import Task, TaskStatus
from src.models.execution import ExecutionResult


class DatabaseManager:
    """SQLite database manager for the automation platform."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self.db_path = Path(config.db_path)
        
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(str(self.db_path))
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
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                parameters TEXT,
                dependencies TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT,
                execution_time REAL,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                tags TEXT,
                priority INTEGER DEFAULT 1,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        """)
        
        # Execution logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_logs (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration REAL DEFAULT 0.0,
                steps TEXT,
                errors TEXT,
                warnings TEXT,
                total_tasks INTEGER DEFAULT 0,
                successful_tasks INTEGER DEFAULT 0,
                failed_tasks INTEGER DEFAULT 0,
                data_extracted TEXT,
                files_processed TEXT,
                api_calls_made INTEGER DEFAULT 0,
                logs TEXT,
                metadata TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        """)
        
        # Audit events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                details TEXT,
                pii_detected BOOLEAN DEFAULT FALSE,
                pii_masked BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                workflow_id TEXT,
                task_id TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES conversations (session_id)
            )
        """)
        
        self.connection.commit()
        self.logger.info("Database tables created successfully")
        
    async def save_workflow(self, workflow: Union[Workflow, Dict[str, Any]]):
        """Save workflow to database."""
        try:
            # Handle both Workflow objects and dictionaries
            if isinstance(workflow, dict):
                # Convert dictionary to Workflow object
                workflow_obj = Workflow(
                    id=workflow.get("id"),
                    name=workflow.get("name", "Unnamed Workflow"),
                    description=workflow.get("description", ""),
                    domain=workflow.get("domain", "general"),
                    status=WorkflowStatus(workflow.get("status", "planning")),
                    created_at=datetime.fromisoformat(workflow.get("created_at")) if workflow.get("created_at") else datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    parameters=workflow.get("parameters", {}),
                    tags=workflow.get("tags", [])
                )
            else:
                workflow_obj = workflow
                
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO workflows 
                (id, name, description, domain, status, parameters, created_at, updated_at, started_at, completed_at, created_by, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_obj.id,
                workflow_obj.name,
                workflow_obj.description,
                workflow_obj.domain,
                workflow_obj.status.value,
                json.dumps(workflow_obj.parameters),
                workflow_obj.created_at.isoformat(),
                workflow_obj.updated_at.isoformat(),
                workflow_obj.started_at.isoformat() if workflow_obj.started_at else None,
                workflow_obj.completed_at.isoformat() if workflow_obj.completed_at else None,
                workflow_obj.created_by,
                json.dumps(workflow_obj.tags)
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
                return Workflow(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    domain=row['domain'],
                    status=WorkflowStatus(row['status']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    created_by=row['created_by'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {},
                    tags=json.loads(row['tags']) if row['tags'] else []
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow: {e}", exc_info=True)
            return None
            
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
                workflow = Workflow(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    domain=row['domain'],
                    status=WorkflowStatus(row['status']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    created_by=row['created_by'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {},
                    tags=json.loads(row['tags']) if row['tags'] else []
                )
                workflows.append(workflow)
                
            return workflows
            
        except Exception as e:
            self.logger.error(f"Failed to get active workflows: {e}", exc_info=True)
            return []
            
    async def save_execution_result(self, execution_result: ExecutionResult):
        """Save execution result to database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO execution_logs 
                (id, workflow_id, success, status, started_at, completed_at, duration, steps, errors, warnings,
                 total_tasks, successful_tasks, failed_tasks, data_extracted, files_processed, api_calls_made, logs, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"exec_{execution_result.workflow_id}_{datetime.utcnow().timestamp()}",
                execution_result.workflow_id,
                execution_result.success,
                execution_result.status.value,
                execution_result.started_at.isoformat(),
                execution_result.completed_at.isoformat() if execution_result.completed_at else None,
                execution_result.duration,
                json.dumps(execution_result.steps),
                json.dumps(execution_result.errors),
                json.dumps(execution_result.warnings),
                execution_result.total_tasks,
                execution_result.successful_tasks,
                execution_result.failed_tasks,
                json.dumps(execution_result.data_extracted),
                json.dumps(execution_result.files_processed),
                execution_result.api_calls_made,
                json.dumps([log.dict() for log in execution_result.logs]),
                json.dumps(execution_result.metadata)
            ))
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save execution result: {e}", exc_info=True)
            raise
            
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from database."""
        try:
            cursor = self.connection.cursor()
            
            # Get workflow statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_workflows,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_workflows,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_workflows,
                    AVG(CASE WHEN completed_at IS NOT NULL 
                        THEN (julianday(completed_at) - julianday(created_at)) * 86400 
                        ELSE NULL END) as avg_duration
                FROM workflows
            """)
            
            workflow_stats = cursor.fetchone()
            
            # Get execution statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                    AVG(duration) as avg_execution_time,
                    SUM(api_calls_made) as total_api_calls
                FROM execution_logs
            """)
            
            execution_stats = cursor.fetchone()
            
            return {
                "workflows": {
                    "total": workflow_stats['total_workflows'] or 0,
                    "completed": workflow_stats['completed_workflows'] or 0,
                    "failed": workflow_stats['failed_workflows'] or 0,
                    "avg_duration": workflow_stats['avg_duration'] or 0.0
                },
                "executions": {
                    "total": execution_stats['total_executions'] or 0,
                    "successful": execution_stats['successful_executions'] or 0,
                    "avg_time": execution_stats['avg_execution_time'] or 0.0,
                    "total_api_calls": execution_stats['total_api_calls'] or 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
            return {}
            
    async def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")