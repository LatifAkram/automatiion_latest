"""
Audit Logging
============

Comprehensive audit logging system for tracking planning activities,
task executions, and conversations for compliance and governance.
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import uuid


class AuditLogger:
    """Comprehensive audit logging system for compliance and governance."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database connection
        self.db_path = Path(config.data_path) / "audit.db"
        self.connection = None
        
        # Audit categories
        self.audit_categories = {
            "planning": "Workflow planning activities",
            "execution": "Task execution activities", 
            "conversation": "Conversation and chat activities",
            "search": "Search and data gathering activities",
            "extraction": "Data extraction activities",
            "system": "System and configuration activities",
            "security": "Security and access control activities",
            "compliance": "Compliance and governance activities"
        }
        
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
    async def initialize(self):
        """Initialize audit logging system."""
        try:
            # Ensure audit directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            await self._initialize_database()
            
            self.logger.info(f"Audit logging system initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit logging: {e}", exc_info=True)
            raise
            
    async def _initialize_database(self):
        """Initialize audit database with tables."""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            cursor = self.connection.cursor()
            
            # Create audit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    workflow_id TEXT,
                    task_id TEXT,
                    details TEXT,
                    pii_detected BOOLEAN DEFAULT FALSE,
                    pii_masked BOOLEAN DEFAULT FALSE,
                    compliance_level TEXT DEFAULT 'standard',
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create audit trails table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trails (
                    id TEXT PRIMARY KEY,
                    parent_event_id TEXT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_state TEXT,
                    after_state TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (parent_event_id) REFERENCES audit_events (id)
                )
            """)
            
            # Create compliance reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    compliance_status TEXT DEFAULT 'compliant',
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_events (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_workflow ON audit_events (workflow_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_events (pii_detected)")
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit database: {e}", exc_info=True)
            raise
            
    async def log_planning_activity(self, workflow_id: str, plan: Dict[str, Any], 
                                  user_id: str = None, session_id: str = None) -> str:
        """
        Log workflow planning activity.
        
        Args:
            workflow_id: Workflow identifier
            plan: Workflow plan details
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Check for PII in plan
            pii_detected, masked_plan = await self._detect_and_mask_pii(plan)
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "planning",
                "event_type": "workflow_plan_created",
                "user_id": user_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "task_id": None,
                "details": json.dumps(masked_plan),
                "pii_detected": pii_detected,
                "pii_masked": pii_detected,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            # Log audit trail
            await self._log_audit_trail(event_id, "plan_created", None, masked_plan)
            
            self.logger.info(f"Logged planning activity for workflow {workflow_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log planning activity: {e}", exc_info=True)
            return ""
            
    async def log_task_execution(self, task_id: str, workflow_id: str, task_type: str,
                               task_data: Dict[str, Any], execution_result: Dict[str, Any],
                               user_id: str = None, session_id: str = None) -> str:
        """
        Log task execution activity.
        
        Args:
            task_id: Task identifier
            workflow_id: Workflow identifier
            task_type: Type of task
            task_data: Task input data
            execution_result: Task execution result
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Check for PII in task data and result
            task_pii_detected, masked_task_data = await self._detect_and_mask_pii(task_data)
            result_pii_detected, masked_result = await self._detect_and_mask_pii(execution_result)
            
            pii_detected = task_pii_detected or result_pii_detected
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "execution",
                "event_type": f"task_{task_type}_executed",
                "user_id": user_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "task_id": task_id,
                "details": json.dumps({
                    "task_data": masked_task_data,
                    "execution_result": masked_result,
                    "task_type": task_type
                }),
                "pii_detected": pii_detected,
                "pii_masked": pii_detected,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            # Log audit trail
            await self._log_audit_trail(event_id, "task_executed", masked_task_data, masked_result)
            
            self.logger.info(f"Logged task execution for task {task_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log task execution: {e}", exc_info=True)
            return ""
            
    async def log_task_execution(self, task_id: str, workflow_id: str, task_type: str,
                               task_data: Dict[str, Any], execution_result: Dict[str, Any],
                               user_id: str = None, session_id: str = None) -> str:
        """
        Log task execution activity (overloaded method for compatibility).
        """
        return await self.log_task_execution(task_id, workflow_id, task_type, task_data, 
                                           execution_result, user_id, session_id)
            
    async def log_conversation(self, session_id: str, message_type: str, message_content: str,
                             user_id: str = None, workflow_id: str = None) -> str:
        """
        Log conversation activity.
        
        Args:
            session_id: Session identifier
            message_type: Type of message (user, assistant, system)
            message_content: Message content
            user_id: User identifier
            workflow_id: Workflow identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Check for PII in message content
            pii_detected, masked_content = await self._detect_and_mask_pii({"content": message_content})
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "conversation",
                "event_type": f"message_{message_type}",
                "user_id": user_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "task_id": None,
                "details": json.dumps({
                    "message_type": message_type,
                    "content": masked_content,
                    "original_length": len(message_content)
                }),
                "pii_detected": pii_detected,
                "pii_masked": pii_detected,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            self.logger.info(f"Logged conversation message for session {session_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {e}", exc_info=True)
            return ""
            
    async def log_search_activity(self, query: str, sources: List[str], results_count: int,
                                user_id: str = None, session_id: str = None, 
                                workflow_id: str = None) -> str:
        """
        Log search activity.
        
        Args:
            query: Search query
            sources: Search sources used
            results_count: Number of results
            user_id: User identifier
            session_id: Session identifier
            workflow_id: Workflow identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Check for PII in search query
            pii_detected, masked_query = await self._detect_and_mask_pii({"query": query})
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "search",
                "event_type": "search_performed",
                "user_id": user_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "task_id": None,
                "details": json.dumps({
                    "query": masked_query,
                    "sources": sources,
                    "results_count": results_count
                }),
                "pii_detected": pii_detected,
                "pii_masked": pii_detected,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            self.logger.info(f"Logged search activity: {len(sources)} sources, {results_count} results")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log search activity: {e}", exc_info=True)
            return ""
            
    async def log_extraction_activity(self, url: str, content_type: str, fields_extracted: List[str],
                                    user_id: str = None, session_id: str = None,
                                    workflow_id: str = None) -> str:
        """
        Log data extraction activity.
        
        Args:
            url: URL extracted from
            content_type: Type of content extracted
            fields_extracted: List of fields extracted
            user_id: User identifier
            session_id: Session identifier
            workflow_id: Workflow identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "extraction",
                "event_type": "data_extracted",
                "user_id": user_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "task_id": None,
                "details": json.dumps({
                    "url": url,
                    "content_type": content_type,
                    "fields_extracted": fields_extracted,
                    "extraction_time": datetime.utcnow().isoformat()
                }),
                "pii_detected": False,
                "pii_masked": False,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            self.logger.info(f"Logged extraction activity for {url}: {len(fields_extracted)} fields")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log extraction activity: {e}", exc_info=True)
            return ""
            
    async def log_system_activity(self, activity_type: str, details: Dict[str, Any],
                                user_id: str = None) -> str:
        """
        Log system activity.
        
        Args:
            activity_type: Type of system activity
            details: Activity details
            user_id: User identifier
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "system",
                "event_type": activity_type,
                "user_id": user_id,
                "session_id": None,
                "workflow_id": None,
                "task_id": None,
                "details": json.dumps(details),
                "pii_detected": False,
                "pii_masked": False,
                "compliance_level": "standard",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            self.logger.info(f"Logged system activity: {activity_type}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log system activity: {e}", exc_info=True)
            return ""
            
    async def log_security_activity(self, security_event: str, details: Dict[str, Any],
                                  user_id: str = None, severity: str = "medium") -> str:
        """
        Log security activity.
        
        Args:
            security_event: Type of security event
            details: Event details
            user_id: User identifier
            severity: Event severity (low, medium, high, critical)
            
        Returns:
            Audit event ID
        """
        try:
            event_id = str(uuid.uuid4())
            
            event_data = {
                "id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "security",
                "event_type": security_event,
                "user_id": user_id,
                "session_id": None,
                "workflow_id": None,
                "task_id": None,
                "details": json.dumps({
                    **details,
                    "severity": severity,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                "pii_detected": False,
                "pii_masked": False,
                "compliance_level": "high",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self._insert_audit_event(event_data)
            
            self.logger.warning(f"Logged security event: {security_event} (severity: {severity})")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log security activity: {e}", exc_info=True)
            return ""
            
    async def _insert_audit_event(self, event_data: Dict[str, Any]):
        """Insert audit event into database."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO audit_events (
                    id, timestamp, category, event_type, user_id, session_id,
                    workflow_id, task_id, details, pii_detected, pii_masked,
                    compliance_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_data["id"], event_data["timestamp"], event_data["category"],
                event_data["event_type"], event_data["user_id"], event_data["session_id"],
                event_data["workflow_id"], event_data["task_id"], event_data["details"],
                event_data["pii_detected"], event_data["pii_masked"],
                event_data["compliance_level"], event_data["created_at"]
            ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to insert audit event: {e}", exc_info=True)
            raise
            
    async def _log_audit_trail(self, parent_event_id: str, action: str, 
                             before_state: Any, after_state: Any, metadata: Dict[str, Any] = None):
        """Log audit trail entry."""
        try:
            cursor = self.connection.cursor()
            
            trail_data = {
                "id": str(uuid.uuid4()),
                "parent_event_id": parent_event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "before_state": json.dumps(before_state) if before_state else None,
                "after_state": json.dumps(after_state) if after_state else None,
                "metadata": json.dumps(metadata) if metadata else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO audit_trails (
                    id, parent_event_id, timestamp, action, before_state,
                    after_state, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trail_data["id"], trail_data["parent_event_id"], trail_data["timestamp"],
                trail_data["action"], trail_data["before_state"], trail_data["after_state"],
                trail_data["metadata"], trail_data["created_at"]
            ))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit trail: {e}", exc_info=True)
            
    async def _detect_and_mask_pii(self, data: Any) -> Tuple[bool, Any]:
        """
        Detect and mask PII in data.
        
        Args:
            data: Data to check for PII
            
        Returns:
            Tuple of (pii_detected, masked_data)
        """
        try:
            if isinstance(data, str):
                return await self._mask_pii_in_text(data)
            elif isinstance(data, dict):
                return await self._mask_pii_in_dict(data)
            elif isinstance(data, list):
                return await self._mask_pii_in_list(data)
            else:
                return False, data
                
        except Exception as e:
            self.logger.warning(f"Failed to detect/mask PII: {e}")
            return False, data
            
    async def _mask_pii_in_text(self, text: str) -> Tuple[bool, str]:
        """Mask PII in text string."""
        try:
            masked_text = text
            pii_detected = False
            
            for pii_type, pattern in self.pii_patterns.items():
                import re
                matches = re.findall(pattern, text)
                if matches:
                    pii_detected = True
                    for match in matches:
                        if pii_type == "email":
                            masked = f"[EMAIL_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                        elif pii_type == "phone":
                            masked = f"[PHONE_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                        elif pii_type == "ssn":
                            masked = f"[SSN_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                        elif pii_type == "credit_card":
                            masked = f"[CC_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                        elif pii_type == "ip_address":
                            masked = f"[IP_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                        else:
                            masked = f"[PII_{hashlib.md5(match.encode()).hexdigest()[:8]}]"
                            
                        masked_text = masked_text.replace(match, masked)
                        
            return pii_detected, masked_text
            
        except Exception as e:
            self.logger.warning(f"Failed to mask PII in text: {e}")
            return False, text
            
    async def _mask_pii_in_dict(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Mask PII in dictionary."""
        try:
            masked_data = {}
            pii_detected = False
            
            for key, value in data.items():
                if isinstance(value, str):
                    detected, masked = await self._mask_pii_in_text(value)
                    pii_detected = pii_detected or detected
                    masked_data[key] = masked
                elif isinstance(value, dict):
                    detected, masked = await self._mask_pii_in_dict(value)
                    pii_detected = pii_detected or detected
                    masked_data[key] = masked
                elif isinstance(value, list):
                    detected, masked = await self._mask_pii_in_list(value)
                    pii_detected = pii_detected or detected
                    masked_data[key] = masked
                else:
                    masked_data[key] = value
                    
            return pii_detected, masked_data
            
        except Exception as e:
            self.logger.warning(f"Failed to mask PII in dict: {e}")
            return False, data
            
    async def _mask_pii_in_list(self, data: List[Any]) -> Tuple[bool, List[Any]]:
        """Mask PII in list."""
        try:
            masked_data = []
            pii_detected = False
            
            for item in data:
                if isinstance(item, str):
                    detected, masked = await self._mask_pii_in_text(item)
                    pii_detected = pii_detected or detected
                    masked_data.append(masked)
                elif isinstance(item, dict):
                    detected, masked = await self._mask_pii_in_dict(item)
                    pii_detected = pii_detected or detected
                    masked_data.append(masked)
                elif isinstance(item, list):
                    detected, masked = await self._mask_pii_in_list(item)
                    pii_detected = pii_detected or detected
                    masked_data.append(masked)
                else:
                    masked_data.append(item)
                    
            return pii_detected, masked_data
            
        except Exception as e:
            self.logger.warning(f"Failed to mask PII in list: {e}")
            return False, data
            
    async def get_audit_events(self, category: str = None, workflow_id: str = None,
                             user_id: str = None, start_date: str = None, 
                             end_date: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            category: Filter by category
            workflow_id: Filter by workflow ID
            user_id: Filter by user ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
                
            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)
                
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
                
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            events = []
            for row in rows:
                event = {
                    "id": row[0],
                    "timestamp": row[1],
                    "category": row[2],
                    "event_type": row[3],
                    "user_id": row[4],
                    "session_id": row[5],
                    "workflow_id": row[6],
                    "task_id": row[7],
                    "details": json.loads(row[8]) if row[8] else None,
                    "pii_detected": bool(row[9]),
                    "pii_masked": bool(row[10]),
                    "compliance_level": row[11],
                    "created_at": row[12]
                }
                events.append(event)
                
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get audit events: {e}", exc_info=True)
            return []
            
    async def generate_compliance_report(self, report_type: str, period_start: str, 
                                       period_end: str) -> Dict[str, Any]:
        """
        Generate compliance report for specified period.
        
        Args:
            report_type: Type of compliance report
            period_start: Start date (ISO format)
            period_end: End date (ISO format)
            
        Returns:
            Compliance report data
        """
        try:
            cursor = self.connection.cursor()
            
            # Get events for period
            cursor.execute("""
                SELECT category, event_type, COUNT(*) as count,
                       SUM(CASE WHEN pii_detected THEN 1 ELSE 0 END) as pii_events
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY category, event_type
            """, (period_start, period_end))
            
            rows = cursor.fetchall()
            
            # Build report data
            report_data = {
                "report_type": report_type,
                "period_start": period_start,
                "period_end": period_end,
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_events": sum(row[2] for row in rows),
                    "pii_events": sum(row[3] for row in rows),
                    "categories": {}
                },
                "details": []
            }
            
            # Process events by category
            for row in rows:
                category, event_type, count, pii_count = row
                
                if category not in report_data["summary"]["categories"]:
                    report_data["summary"]["categories"][category] = {
                        "total_events": 0,
                        "pii_events": 0,
                        "event_types": {}
                    }
                    
                report_data["summary"]["categories"][category]["total_events"] += count
                report_data["summary"]["categories"][category]["pii_events"] += pii_count
                report_data["summary"]["categories"][category]["event_types"][event_type] = {
                    "count": count,
                    "pii_count": pii_count
                }
                
                report_data["details"].append({
                    "category": category,
                    "event_type": event_type,
                    "count": count,
                    "pii_count": pii_count
                })
                
            # Determine compliance status
            total_events = report_data["summary"]["total_events"]
            pii_events = report_data["summary"]["pii_events"]
            
            if pii_events == 0:
                compliance_status = "compliant"
            elif pii_events / total_events < 0.01:  # Less than 1% PII events
                compliance_status = "mostly_compliant"
            else:
                compliance_status = "non_compliant"
                
            report_data["compliance_status"] = compliance_status
            
            # Save report to database
            report_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO compliance_reports (
                    id, report_type, period_start, period_end, generated_at,
                    report_data, compliance_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id, report_type, period_start, period_end,
                report_data["generated_at"], json.dumps(report_data),
                compliance_status, datetime.utcnow().isoformat()
            ))
            
            self.connection.commit()
            
            self.logger.info(f"Generated compliance report: {report_type} ({compliance_status})")
            return report_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}", exc_info=True)
            return {}
            
    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        try:
            cursor = self.connection.cursor()
            
            # Get total events
            cursor.execute("SELECT COUNT(*) FROM audit_events")
            total_events = cursor.fetchone()[0]
            
            # Get events by category
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM audit_events 
                GROUP BY category
            """)
            category_counts = dict(cursor.fetchall())
            
            # Get PII events
            cursor.execute("SELECT COUNT(*) FROM audit_events WHERE pii_detected = 1")
            pii_events = cursor.fetchone()[0]
            
            # Get recent events (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp >= datetime('now', '-1 day')
            """)
            recent_events = cursor.fetchone()[0]
            
            return {
                "total_events": total_events,
                "category_counts": category_counts,
                "pii_events": pii_events,
                "recent_events_24h": recent_events,
                "compliance_levels": {
                    "standard": total_events - pii_events,
                    "high": pii_events
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get audit statistics: {e}", exc_info=True)
            return {}
            
    async def shutdown(self):
        """Shutdown audit logging system."""
        try:
            if self.connection:
                self.connection.close()
                
            self.logger.info("Audit logging system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during audit logging shutdown: {e}", exc_info=True)