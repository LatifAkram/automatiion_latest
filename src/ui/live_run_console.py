#!/usr/bin/env python3
"""
SUPER-OMEGA Live Run Console
============================

Complete live console with web interface, chat, step tiles, and crash recovery.
This addresses the critical missing UI component for real-time automation monitoring.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import websockets
import aiofiles
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, WebSocket, HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError:
    print("⚠️  FastAPI not available - using minimal implementation")
    FastAPI = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RunStep:
    """Individual automation step"""
    step_id: str
    timestamp: str
    description: str
    status: str  # 'pending', 'running', 'success', 'failed', 'retrying'
    duration_ms: float
    retries: int
    confidence: float
    screenshot_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class LiveRun:
    """Complete automation run"""
    run_id: str
    start_time: str
    end_time: Optional[str]
    status: str  # 'running', 'completed', 'failed', 'crashed'
    goal: str
    steps: List[RunStep]
    artifacts: List[str]
    recovery_point: Optional[str] = None
    total_steps: int = 0
    success_rate: float = 0.0

class LiveRunConsole:
    """Complete Live Run Console with web interface"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.active_runs: Dict[str, LiveRun] = {}
        self.connected_clients: set = set()
        self.console_history: List[Dict[str, Any]] = []
        
        # Create directories
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)
        
        self.static_dir = Path("static")
        self.static_dir.mkdir(exist_ok=True)
        
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize web assets
        asyncio.create_task(self.create_web_assets())
    
    async def create_web_assets(self):
        """Create HTML templates and static files"""
        
        # Main HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUPER-OMEGA Live Console</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Monaco', 'Consolas', monospace; 
            background: #0a0a0a; 
            color: #00ff00; 
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .sidebar { 
            width: 300px; 
            background: #1a1a1a; 
            border-right: 2px solid #333; 
            padding: 20px;
        }
        .main-content { flex: 1; display: flex; flex-direction: column; }
        .header { 
            background: #2a2a2a; 
            padding: 15px 20px; 
            border-bottom: 2px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .run-area { flex: 1; display: flex; }
        .steps-panel { 
            flex: 1; 
            padding: 20px; 
            overflow-y: auto; 
            background: #0f0f0f;
        }
        .chat-panel { 
            width: 350px; 
            background: #1a1a1a; 
            border-left: 2px solid #333;
            display: flex;
            flex-direction: column;
        }
        .step-tile { 
            background: #1a1a1a; 
            border: 1px solid #333; 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        .step-tile.running { border-color: #ffaa00; box-shadow: 0 0 10px rgba(255,170,0,0.3); }
        .step-tile.success { border-color: #00ff00; }
        .step-tile.failed { border-color: #ff0000; }
        .step-tile.retrying { border-color: #ff8800; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .step-header { display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }
        .step-status { 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 12px; 
            font-weight: bold;
        }
        .status-pending { background: #666; }
        .status-running { background: #ffaa00; color: #000; }
        .status-success { background: #00ff00; color: #000; }
        .status-failed { background: #ff0000; }
        .status-retrying { background: #ff8800; color: #000; }
        .screenshot { 
            max-width: 100%; 
            border: 1px solid #333; 
            border-radius: 4px; 
            margin-top: 10px;
        }
        .chat-messages { 
            flex: 1; 
            padding: 20px; 
            overflow-y: auto; 
            border-bottom: 2px solid #333;
        }
        .chat-input { 
            padding: 20px; 
            display: flex; 
            gap: 10px;
        }
        .chat-input input { 
            flex: 1; 
            background: #0a0a0a; 
            border: 1px solid #333; 
            color: #00ff00; 
            padding: 10px; 
            border-radius: 4px;
        }
        .chat-input button { 
            background: #00ff00; 
            color: #000; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 4px; 
            cursor: pointer;
        }
        .message { 
            margin-bottom: 15px; 
            padding: 10px; 
            border-radius: 6px;
        }
        .message.user { background: #2a2a2a; }
        .message.system { background: #1a3a1a; }
        .message.error { background: #3a1a1a; }
        .run-controls { display: flex; gap: 10px; }
        .btn { 
            padding: 8px 16px; 
            border: 1px solid #333; 
            background: #2a2a2a; 
            color: #00ff00; 
            border-radius: 4px; 
            cursor: pointer;
        }
        .btn:hover { background: #3a3a3a; }
        .btn.primary { background: #00ff00; color: #000; }
        .metrics { display: flex; gap: 20px; font-size: 14px; }
        .metric { text-align: center; }
        .metric-value { font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🚀 SUPER-OMEGA</h2>
            <h3>Live Console</h3>
            <div style="margin-top: 20px;">
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="activeRuns">0</div>
                        <div>Active</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="successRate">0%</div>
                        <div>Success</div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 30px;">
                <h4>Recent Runs</h4>
                <div id="recentRuns"></div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1 id="currentRunTitle">No Active Run</h1>
                <div class="run-controls">
                    <button class="btn" onclick="pauseRun()">⏸️ Pause</button>
                    <button class="btn" onclick="resumeRun()">▶️ Resume</button>
                    <button class="btn" onclick="stopRun()">⏹️ Stop</button>
                    <button class="btn primary" onclick="startNewRun()">🚀 New Run</button>
                </div>
            </div>
            
            <div class="run-area">
                <div class="steps-panel">
                    <div id="stepsContainer">
                        <div style="text-align: center; color: #666; margin-top: 100px;">
                            <h3>No active automation run</h3>
                            <p>Start a new run to see live step execution</p>
                        </div>
                    </div>
                </div>
                
                <div class="chat-panel">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message system">
                            <strong>SUPER-OMEGA Console</strong><br>
                            Ready for automation. Type commands or questions below.
                        </div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="chatInput" placeholder="Type a command or question..." 
                               onkeypress="if(event.key==='Enter') sendMessage()">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentRunId = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('Connected to SUPER-OMEGA console');
                addChatMessage('system', 'Connected to live console');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('Disconnected from console');
                addChatMessage('error', 'Connection lost - attempting reconnect...');
                setTimeout(connectWebSocket, 3000);
            };
        }

        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'run_update':
                    updateRunDisplay(data.run);
                    break;
                case 'step_update':
                    updateStepTile(data.step);
                    break;
                case 'chat_response':
                    addChatMessage('system', data.message);
                    break;
                case 'metrics_update':
                    updateMetrics(data.metrics);
                    break;
            }
        }

        function updateRunDisplay(run) {
            currentRunId = run.run_id;
            document.getElementById('currentRunTitle').textContent = 
                `Run: ${run.goal} (${run.status})`;
            
            const container = document.getElementById('stepsContainer');
            container.innerHTML = '';
            
            run.steps.forEach(step => {
                container.appendChild(createStepTile(step));
            });
        }

        function createStepTile(step) {
            const tile = document.createElement('div');
            tile.className = `step-tile ${step.status}`;
            tile.id = `step-${step.step_id}`;
            
            tile.innerHTML = `
                <div class="step-header">
                    <span>${step.description}</span>
                    <span class="step-status status-${step.status}">${step.status.toUpperCase()}</span>
                </div>
                <div style="font-size: 12px; color: #888;">
                    Duration: ${step.duration_ms.toFixed(2)}ms | 
                    Retries: ${step.retries} | 
                    Confidence: ${(step.confidence * 100).toFixed(1)}%
                </div>
                ${step.screenshot_path ? `<img src="/screenshots/${step.screenshot_path}" class="screenshot">` : ''}
                ${step.error_message ? `<div style="color: #ff6666; margin-top: 10px;">${step.error_message}</div>` : ''}
            `;
            
            return tile;
        }

        function updateStepTile(step) {
            const existing = document.getElementById(`step-${step.step_id}`);
            if (existing) {
                existing.replaceWith(createStepTile(step));
            }
        }

        function addChatMessage(type, message) {
            const container = document.getElementById('chatMessages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${type}`;
            msgDiv.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong><br>${message}`;
            container.appendChild(msgDiv);
            container.scrollTop = container.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            addChatMessage('user', message);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'chat_message',
                    message: message,
                    run_id: currentRunId
                }));
            }
            
            input.value = '';
        }

        function updateMetrics(metrics) {
            document.getElementById('activeRuns').textContent = metrics.active_runs;
            document.getElementById('successRate').textContent = `${metrics.success_rate}%`;
        }

        function startNewRun() {
            const goal = prompt('Enter automation goal:');
            if (goal && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'start_run',
                    goal: goal
                }));
            }
        }

        function pauseRun() {
            if (currentRunId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'pause_run',
                    run_id: currentRunId
                }));
            }
        }

        function resumeRun() {
            if (currentRunId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'resume_run',
                    run_id: currentRunId
                }));
            }
        }

        function stopRun() {
            if (currentRunId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop_run',
                    run_id: currentRunId
                }));
            }
        }

        // Initialize
        connectWebSocket();
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'ping'}));
            }
        }, 5000);
    </script>
</body>
</html>
        """
        
        # Save HTML template
        async with aiofiles.open(self.templates_dir / "console.html", "w") as f:
            await f.write(html_template)
    
    async def start_run(self, goal: str) -> str:
        """Start a new automation run"""
        run_id = str(uuid.uuid4())
        
        run = LiveRun(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            status="running",
            goal=goal,
            steps=[],
            artifacts=[]
        )
        
        self.active_runs[run_id] = run
        
        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "steps").mkdir(exist_ok=True)
        (run_dir / "frames").mkdir(exist_ok=True)
        
        logger.info(f"🚀 Started new run: {run_id} - {goal}")
        
        # Broadcast to all connected clients
        await self.broadcast_update({
            "type": "run_update",
            "run": asdict(run)
        })
        
        return run_id
    
    async def add_step(self, run_id: str, description: str, 
                      status: str = "pending") -> str:
        """Add a step to a running automation"""
        if run_id not in self.active_runs:
            raise ValueError(f"Run {run_id} not found")
        
        step_id = str(uuid.uuid4())
        step = RunStep(
            step_id=step_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            status=status,
            duration_ms=0.0,
            retries=0,
            confidence=1.0,
            metadata={}
        )
        
        self.active_runs[run_id].steps.append(step)
        
        # Save step to file
        run_dir = self.runs_dir / run_id / "steps"
        step_file = run_dir / f"{len(self.active_runs[run_id].steps):04d}.json"
        
        async with aiofiles.open(step_file, "w") as f:
            await f.write(json.dumps(asdict(step), indent=2))
        
        await self.broadcast_update({
            "type": "step_update",
            "step": asdict(step)
        })
        
        return step_id
    
    async def update_step(self, run_id: str, step_id: str, 
                         status: str, duration_ms: float = 0.0,
                         error_message: str = None, retries: int = 0):
        """Update step status"""
        if run_id not in self.active_runs:
            return
        
        run = self.active_runs[run_id]
        for step in run.steps:
            if step.step_id == step_id:
                step.status = status
                step.duration_ms = duration_ms
                step.retries = retries
                if error_message:
                    step.error_message = error_message
                break
        
        await self.broadcast_update({
            "type": "step_update",
            "step": asdict(step)
        })
    
    async def complete_run(self, run_id: str, status: str = "completed"):
        """Complete an automation run"""
        if run_id not in self.active_runs:
            return
        
        run = self.active_runs[run_id]
        run.end_time = datetime.now().isoformat()
        run.status = status
        run.total_steps = len(run.steps)
        
        # Calculate success rate
        successful_steps = sum(1 for step in run.steps if step.status == "success")
        run.success_rate = (successful_steps / run.total_steps * 100) if run.total_steps > 0 else 0
        
        # Generate final report
        report = {
            "run_id": run_id,
            "goal": run.goal,
            "status": status,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "total_steps": run.total_steps,
            "success_rate": run.success_rate,
            "steps": [asdict(step) for step in run.steps]
        }
        
        # Save report
        report_file = self.runs_dir / run_id / "report.json"
        async with aiofiles.open(report_file, "w") as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"✅ Completed run {run_id}: {status} ({run.success_rate:.1f}% success)")
        
        await self.broadcast_update({
            "type": "run_update",
            "run": asdict(run)
        })
    
    async def handle_crash_recovery(self, run_id: str) -> Optional[str]:
        """Handle crash recovery for a run"""
        if run_id not in self.active_runs:
            return None
        
        run = self.active_runs[run_id]
        
        # Find last successful step
        last_success = None
        for step in reversed(run.steps):
            if step.status == "success":
                last_success = step
                break
        
        if last_success:
            run.recovery_point = last_success.step_id
            logger.info(f"🔄 Recovery point set for run {run_id}: {last_success.description}")
            return last_success.step_id
        
        return None
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if self.connected_clients:
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(json.dumps(message))
                except:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected
    
    async def handle_chat_message(self, message: str, run_id: str = None) -> str:
        """Handle chat messages and return response"""
        
        # Simple command processing
        message_lower = message.lower()
        
        if "status" in message_lower:
            if run_id and run_id in self.active_runs:
                run = self.active_runs[run_id]
                return f"Run {run_id}: {run.status} - {len(run.steps)} steps completed"
            else:
                return f"No active run. Total runs: {len(self.active_runs)}"
        
        elif "help" in message_lower:
            return """Available commands:
• status - Get current run status
• pause - Pause current run
• resume - Resume paused run  
• stop - Stop current run
• runs - List all runs
• help - Show this help"""
        
        elif "runs" in message_lower:
            if self.active_runs:
                runs_info = []
                for run_id, run in self.active_runs.items():
                    runs_info.append(f"• {run_id[:8]}: {run.goal} ({run.status})")
                return "Active runs:\n" + "\n".join(runs_info)
            else:
                return "No active runs"
        
        else:
            return f"Received: {message}. Type 'help' for available commands."
    
    def create_fastapi_app(self) -> Optional[Any]:
        """Create FastAPI application if available"""
        if not FastAPI:
            return None
        
        app = FastAPI(title="SUPER-OMEGA Live Console")
        
        @app.get("/", response_class=HTMLResponse)
        async def get_console():
            async with aiofiles.open(self.templates_dir / "console.html") as f:
                return await f.read()
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket):
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "chat_message":
                        response = await self.handle_chat_message(
                            message["message"], 
                            message.get("run_id")
                        )
                        await websocket.send_text(json.dumps({
                            "type": "chat_response",
                            "message": response
                        }))
                    
                    elif message["type"] == "start_run":
                        run_id = await self.start_run(message["goal"])
                        await websocket.send_text(json.dumps({
                            "type": "chat_response",
                            "message": f"Started new run: {run_id}"
                        }))
                    
                    elif message["type"] == "ping":
                        # Send metrics update
                        active_count = len([r for r in self.active_runs.values() 
                                          if r.status == "running"])
                        success_rates = [r.success_rate for r in self.active_runs.values() 
                                       if r.success_rate > 0]
                        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
                        
                        await websocket.send_text(json.dumps({
                            "type": "metrics_update",
                            "metrics": {
                                "active_runs": active_count,
                                "success_rate": round(avg_success, 1)
                            }
                        }))
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.connected_clients.discard(websocket)
        
        return app
    
    async def start_server(self):
        """Start the live console server"""
        app = self.create_fastapi_app()
        
        if app:
            logger.info(f"🌐 Starting SUPER-OMEGA Live Console at http://{self.host}:{self.port}")
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        else:
            logger.error("❌ FastAPI not available - cannot start web console")
            # Fallback to console-only mode
            await self.console_mode()
    
    async def console_mode(self):
        """Fallback console mode without web interface"""
        logger.info("📟 Running in console-only mode")
        
        while True:
            try:
                command = input("SUPER-OMEGA> ").strip()
                if command.lower() in ['quit', 'exit']:
                    break
                elif command.lower().startswith('start '):
                    goal = command[6:]
                    run_id = await self.start_run(goal)
                    print(f"Started run: {run_id}")
                elif command.lower() == 'status':
                    if self.active_runs:
                        for run_id, run in self.active_runs.items():
                            print(f"Run {run_id[:8]}: {run.goal} ({run.status})")
                    else:
                        print("No active runs")
                else:
                    response = await self.handle_chat_message(command)
                    print(response)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        logger.info("👋 Console shutting down")

async def main():
    """Start the Live Run Console"""
    console = LiveRunConsole()
    
    print("🚀 SUPER-OMEGA Live Run Console")
    print("=" * 50)
    print("Starting live automation monitoring console...")
    print()
    
    # Demo run for testing
    demo_run_id = await console.start_run("Demo: Login to Gmail")
    
    # Add some demo steps
    step1_id = await console.add_step(demo_run_id, "Navigate to gmail.com", "running")
    await asyncio.sleep(1)
    await console.update_step(demo_run_id, step1_id, "success", 1250.0)
    
    step2_id = await console.add_step(demo_run_id, "Find email input field", "running")
    await asyncio.sleep(1.5)
    await console.update_step(demo_run_id, step2_id, "success", 850.0)
    
    step3_id = await console.add_step(demo_run_id, "Enter email address", "running")
    await asyncio.sleep(0.8)
    await console.update_step(demo_run_id, step3_id, "success", 650.0)
    
    await console.complete_run(demo_run_id, "completed")
    
    # Start the server
    await console.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Live Console stopped by user")
    except Exception as e:
        print(f"❌ Console error: {e}")
        logger.error(f"Console startup failed: {e}")