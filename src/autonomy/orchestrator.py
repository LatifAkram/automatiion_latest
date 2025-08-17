#!/usr/bin/env python3
"""
Autonomous Orchestrator
- Polls JobStore for pending jobs
- Executes workflow jobs (navigate/click/type/select/upload/wait/assert/scroll)
- Sends webhook notifications on completion/failure
- Resumable: job status tracked in SQLite
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib import request

from pathlib import Path

from autonomy.job_store import JobStore

# Import live automation (console uses the same backend)
from testing.super_omega_live_automation_fixed import (
    get_fixed_super_omega_live_automation,
    ExecutionMode
)

class AutonomousOrchestrator:
    def __init__(self, poll_interval: float = 1.0):
        self.store = JobStore()
        self.poll_interval = poll_interval
        self._stop = False
        self._automation = None

    async def _ensure_automation(self):
        if self._automation is None:
            self._automation = get_fixed_super_omega_live_automation({'headless': True})

    def submit_workflow(self, steps: List[Dict[str, Any]], priority: int = 0, run_at: Optional[float] = None, webhook: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())
        job = self.store.create_job(job_id, 'workflow', {'steps': steps}, priority=priority, run_at=run_at)
        if webhook:
            self.store.add_webhook(job_id, webhook, 'completed')
            self.store.add_webhook(job_id, webhook, 'failed')
        return job_id

    async def run_forever(self):
        await self._ensure_automation()
        while not self._stop:
            pending = self.store.list_pending_jobs(limit=5)
            if not pending:
                await asyncio.sleep(self.poll_interval)
                continue
            for job in pending:
                await self._run_job(job.id, job.params)
            await asyncio.sleep(0)  # yield

    async def _run_job(self, job_id: str, params: Dict[str, Any]):
        self.store.update_job_status(job_id, 'running')
        steps: List[Dict[str, Any]] = params.get('steps', [])
        session_id = f"job_{job_id[:8]}_{int(time.time())}"
        try:
            await self._ensure_automation()
            await self._automation.create_super_omega_session(session_id, 'about:blank', ExecutionMode.HYBRID)
            for idx, step in enumerate(steps):
                self.store.add_step(job_id, idx, step.get('action',''), step)
                try:
                    res = await self._execute_step(session_id, step)
                    self.store.finish_step(job_id, idx, 'completed' if res.get('success') else 'failed', res)
                    if not res.get('success'):
                        raise RuntimeError(res.get('error','step failed'))
                except Exception as e:
                    self.store.finish_step(job_id, idx, 'failed', {'error': str(e)})
                    raise
            await self._automation.close_super_omega_session(session_id)
            self.store.update_job_status(job_id, 'completed')
            await self._notify(job_id, 'completed')
        except Exception as e:
            try:
                await self._automation.close_super_omega_session(session_id)
            except Exception:
                pass
            self.store.update_job_status(job_id, 'failed', last_error=str(e))
            await self._notify(job_id, 'failed')

    async def _execute_step(self, session_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        a = step.get('action','').lower()
        if a == 'navigate':
            return await self._automation.super_omega_navigate(session_id, step['url'])
        if a == 'find':
            return await self._automation.super_omega_find_element(session_id, step['selector'])
        if a == 'click':
            return await self._automation.super_omega_click(session_id, step['selector'])
        if a == 'type':
            return await self._automation.super_omega_type(session_id, step['selector'], step.get('text',''))
        if a == 'select':
            return await self._automation.super_omega_select_option(session_id, step['selector'], step['value'])
        if a == 'upload':
            return await self._automation.super_omega_upload_file(session_id, step['selector'], step['file'])
        if a == 'wait_for_selector':
            return await self._automation.super_omega_wait_for_selector(session_id, step['selector'], state=step.get('state','visible'), timeout=int(step.get('timeout_ms', 10000)))
        if a == 'assert_text':
            return await self._automation.super_omega_assert_text(session_id, step['selector'], step['contains'])
        if a == 'scroll':
            return await self._automation.super_omega_scroll_to(session_id, step['selector'])
        if a == 'wait':
            await asyncio.sleep(int(step.get('ms', 1000))/1000.0)
            return {'success': True, 'action':'wait'}
        raise ValueError(f"Unsupported action: {a}")

    async def _notify(self, job_id: str, event: str):
        urls = self.store.list_webhooks(job_id, event)
        for url in urls:
            try:
                data = json.dumps({'job_id': job_id, 'event': event, 'timestamp': time.time()}).encode('utf-8')
                req = request.Request(url, data=data, headers={'Content-Type':'application/json'})
                request.urlopen(req, timeout=3)
            except Exception:
                continue

if __name__ == '__main__':
    async def main():
        orch = AutonomousOrchestrator()
        await orch.run_forever()
    asyncio.run(main())