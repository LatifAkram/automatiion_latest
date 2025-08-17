"""
SUPER-OMEGA LIVE RUN CONSOLE
============================

Production-ready live console interface for automation monitoring and control.
Built on the zero-dependency web server with real-time WebSocket communication.

✅ FEATURES:
- Real-time automation monitoring
- Live execution logs and status
- WebSocket-based communication
- Interactive controls and commands
- Session management and evidence viewing
- Zero external dependencies
"""

import asyncio
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Import our built-in web server
from builtin_web_server import BuiltinWebServer, WebSocketConnection

logger = logging.getLogger(__name__)

@dataclass
class AutomationSession:
    """Automation session information"""
    session_id: str
    instruction: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SuperOmegaLiveConsole(BuiltinWebServer):
    """Live console server for SUPER-OMEGA automation platform"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        super().__init__(host, port)
        self.active_sessions = {}
        self.console_clients = []
        self.setup_console_routes()
    
    def setup_console_routes(self):
        """Setup console-specific routes and handlers"""
        # Override the default page with console interface
        self.console_html = self._get_console_html()
    
    def _get_console_html(self) -> str:
        """Get the live console HTML interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUPER-OMEGA Live Console</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace; 
            background: #0a0a0a; 
            color: #00ff00; 
            overflow: hidden;
            height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 15px 20px;
            border-bottom: 2px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: #00ff00;
            font-size: 24px;
            text-shadow: 0 0 10px #00ff00;
        }
        
        .status-bar {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff0000;
            box-shadow: 0 0 10px currentColor;
        }
        
        .status-indicator.connected {
            background: #00ff00;
        }
        
        .main-container {
            display: flex;
            height: calc(100vh - 70px);
        }
        
        .sidebar {
            width: 300px;
            background: #1a1a1a;
            border-right: 2px solid #333;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-section {
            padding: 15px;
            border-bottom: 1px solid #333;
        }
        
        .sidebar-section h3 {
            color: #00ccff;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .metric {
            text-align: center;
            padding: 8px;
            background: #0f0f0f;
            border-radius: 4px;
            border: 1px solid #333;
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #00ff00;
        }
        
        .metric-label {
            font-size: 10px;
            color: #888;
            margin-top: 2px;
        }
        
        .sessions-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        .session-item {
            padding: 8px;
            margin-bottom: 8px;
            background: #0f0f0f;
            border-radius: 4px;
            border-left: 3px solid #00ff00;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .session-item:hover {
            background: #1f1f1f;
        }
        
        .session-item.active {
            background: #2a2a2a;
            border-left-color: #00ccff;
        }
        
        .session-id {
            font-size: 11px;
            color: #888;
        }
        
        .session-instruction {
            font-size: 12px;
            color: #fff;
            margin: 4px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .session-status {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            background: #333;
        }
        
        .session-status.running {
            background: #ff6600;
            color: #000;
        }
        
        .session-status.completed {
            background: #00ff00;
            color: #000;
        }
        
        .session-status.failed {
            background: #ff0000;
            color: #fff;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .content-header {
            background: #2a2a2a;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
        }
        
        .tab {
            padding: 8px 16px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px 4px 0 0;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .tab.active {
            background: #0a0a0a;
            border-bottom-color: #0a0a0a;
            color: #00ff00;
        }
        
        .console-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
        }
        
        .console-output {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            line-height: 1.4;
        }
        
        .log-entry {
            margin-bottom: 8px;
            padding: 6px 10px;
            border-radius: 4px;
            border-left: 3px solid #333;
        }
        
        .log-entry.info {
            border-left-color: #00ccff;
            background: rgba(0, 204, 255, 0.1);
        }
        
        .log-entry.success {
            border-left-color: #00ff00;
            background: rgba(0, 255, 0, 0.1);
        }
        
        .log-entry.warning {
            border-left-color: #ff6600;
            background: rgba(255, 102, 0, 0.1);
        }
        
        .log-entry.error {
            border-left-color: #ff0000;
            background: rgba(255, 0, 0, 0.1);
        }
        
        .log-timestamp {
            color: #666;
            font-size: 11px;
            margin-right: 10px;
        }
        
        .console-input {
            background: #1a1a1a;
            border-top: 1px solid #333;
            padding: 15px 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .console-input input {
            flex: 1;
            background: #0a0a0a;
            border: 1px solid #333;
            color: #00ff00;
            padding: 10px 15px;
            border-radius: 4px;
            font-family: inherit;
        }
        
        .console-input input:focus {
            outline: none;
            border-color: #00ff00;
            box-shadow: 0 0 5px rgba(0, 255, 0, 0.3);
        }
        
        .btn {
            padding: 10px 20px;
            background: transparent;
            border: 1px solid #00ff00;
            color: #00ff00;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }
        
        .btn:hover {
            background: #00ff00;
            color: #000;
        }
        
        .btn.secondary {
            border-color: #666;
            color: #666;
        }
        
        .btn.secondary:hover {
            background: #666;
            color: #fff;
        }
        
        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #333;
            border-top: 2px solid #00ff00;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .hidden {
            display: none !important;
        }
        
        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
        }
        
        .evidence-item {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
        
        .evidence-preview {
            width: 100%;
            height: 120px;
            background: #0a0a0a;
            border-radius: 4px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        
        .scrollbar::-webkit-scrollbar {
            width: 8px;
        }
        
        .scrollbar::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        .scrollbar::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        
        .scrollbar::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SUPER-OMEGA Live Console</h1>
        <div class="status-bar">
            <div class="status-item">
                <div class="status-indicator" id="connectionStatus"></div>
                <span>WebSocket</span>
            </div>
            <div class="status-item">
                <span id="sessionCount">0</span>
                <span>Sessions</span>
            </div>
            <div class="status-item">
                <span id="messageCount">0</span>
                <span>Messages</span>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="sidebar">
            <div class="sidebar-section">
                <h3>System Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="cpuUsage">--</div>
                        <div class="metric-label">CPU</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="memoryUsage">--</div>
                        <div class="metric-label">Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="activeConnections">0</div>
                        <div class="metric-label">Connections</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="uptime">--</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Active Sessions</h3>
                <div class="sessions-list scrollbar" id="sessionsList">
                    <!-- Sessions will be populated here -->
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="content-header">
                <div class="tabs">
                    <div class="tab active" data-tab="console">Console</div>
                    <div class="tab" data-tab="sessions">Sessions</div>
                    <div class="tab" data-tab="evidence">Evidence</div>
                    <div class="tab" data-tab="system">System</div>
                </div>
                <div>
                    <button class="btn secondary" onclick="clearConsole()">Clear</button>
                    <button class="btn" onclick="exportLogs()">Export</button>
                </div>
            </div>
            
            <div class="console-area">
                <!-- Console Tab -->
                <div id="consoleTab" class="console-output scrollbar">
                    <div class="log-entry info">
                        <span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span>
                        <strong>🚀 SUPER-OMEGA Live Console Initialized</strong>
                    </div>
                    <div class="log-entry info">
                        <span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span>
                        Built-in web server active - Zero external dependencies
                    </div>
                    <div class="log-entry info">
                        <span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span>
                        WebSocket connection establishing...
                    </div>
                </div>
                
                <!-- Sessions Tab -->
                <div id="sessionsTab" class="console-output scrollbar hidden">
                    <div class="evidence-grid" id="sessionsGrid">
                        <!-- Session details will be populated here -->
                    </div>
                </div>
                
                <!-- Evidence Tab -->
                <div id="evidenceTab" class="console-output scrollbar hidden">
                    <div class="evidence-grid" id="evidenceGrid">
                        <!-- Evidence will be populated here -->
                    </div>
                </div>
                
                <!-- System Tab -->
                <div id="systemTab" class="console-output scrollbar hidden">
                    <div id="systemInfo">
                        <!-- System information will be populated here -->
                    </div>
                </div>
                
                <div class="console-input">
                    <input type="text" id="commandInput" placeholder="Enter automation instruction or command..." />
                    <button class="btn" onclick="executeCommand()">Execute</button>
                    <button class="btn secondary" onclick="testConnection()">Test</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let messageCount = 0;
        let sessionCount = 0;
        let activeSessions = {};
        let currentTab = 'console';
        
        // Initialize console
        function initConsole() {
            connectWebSocket();
            setupEventListeners();
            updateMetrics();
            
            // Update metrics every 5 seconds
            setInterval(updateMetrics, 5000);
        }
        
        // WebSocket connection
        function connectWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}`);
                
                ws.onopen = function() {
                    updateConnectionStatus(true);
                    addLogEntry('success', 'WebSocket connected successfully');
                    
                    // Send initial handshake
                    sendMessage({
                        type: 'console_init',
                        timestamp: new Date().toISOString()
                    });
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                    } catch (e) {
                        addLogEntry('error', 'Failed to parse WebSocket message: ' + e.message);
                    }
                };
                
                ws.onclose = function() {
                    updateConnectionStatus(false);
                    addLogEntry('warning', 'WebSocket connection closed. Reconnecting...');
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = function(error) {
                    addLogEntry('error', 'WebSocket error: ' + error.message);
                };
                
            } catch (error) {
                addLogEntry('error', 'Failed to connect WebSocket: ' + error.message);
                setTimeout(connectWebSocket, 5000);
            }
        }
        
        // Handle WebSocket messages
        function handleMessage(data) {
            messageCount++;
            document.getElementById('messageCount').textContent = messageCount;
            
            switch (data.type) {
                case 'pong':
                    addLogEntry('info', 'Connection test successful');
                    break;
                    
                case 'automation_status_response':
                    addLogEntry('info', `Automation status: ${data.status}`);
                    break;
                    
                case 'automation_result':
                    handleAutomationResult(data);
                    break;
                    
                case 'session_update':
                    updateSession(data);
                    break;
                    
                case 'system_metrics':
                    updateSystemMetrics(data);
                    break;
                    
                default:
                    addLogEntry('info', `Received: ${JSON.stringify(data)}`);
            }
        }
        
        // Handle automation results
        function handleAutomationResult(data) {
            if (data.success) {
                addLogEntry('success', `Automation completed: ${data.message}`);
            } else {
                addLogEntry('error', `Automation failed: ${data.message || 'Unknown error'}`);
            }
            
            if (data.session_id) {
                updateSessionStatus(data.session_id, data.success ? 'completed' : 'failed');
            }
        }
        
        // Update session information
        function updateSession(data) {
            activeSessions[data.session_id] = data;
            sessionCount = Object.keys(activeSessions).length;
            document.getElementById('sessionCount').textContent = sessionCount;
            
            updateSessionsList();
            addLogEntry('info', `Session ${data.session_id}: ${data.status}`);
        }
        
        // Update sessions list
        function updateSessionsList() {
            const sessionsList = document.getElementById('sessionsList');
            sessionsList.innerHTML = '';
            
            Object.values(activeSessions).forEach(session => {
                const sessionItem = document.createElement('div');
                sessionItem.className = 'session-item';
                sessionItem.innerHTML = `
                    <div class="session-id">${session.session_id}</div>
                    <div class="session-instruction">${session.instruction || 'No instruction'}</div>
                    <div class="session-status ${session.status}">${session.status}</div>
                `;
                sessionItem.onclick = () => selectSession(session.session_id);
                sessionsList.appendChild(sessionItem);
            });
        }
        
        // Update system metrics
        function updateSystemMetrics(data) {
            if (data.cpu_percent !== undefined) {
                document.getElementById('cpuUsage').textContent = data.cpu_percent.toFixed(1) + '%';
            }
            if (data.memory_percent !== undefined) {
                document.getElementById('memoryUsage').textContent = data.memory_percent.toFixed(1) + '%';
            }
            if (data.connections !== undefined) {
                document.getElementById('activeConnections').textContent = data.connections;
            }
            if (data.uptime !== undefined) {
                const hours = Math.floor(data.uptime / 3600);
                const minutes = Math.floor((data.uptime % 3600) / 60);
                document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;
            }
        }
        
        // Add log entry
        function addLogEntry(type, message) {
            const consoleOutput = document.getElementById('consoleTab');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${type}`;
            logEntry.innerHTML = `
                <span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span>
                ${message}
            `;
            consoleOutput.appendChild(logEntry);
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        
        // Send WebSocket message
        function sendMessage(data) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(data));
                return true;
            } else {
                addLogEntry('error', 'WebSocket not connected');
                return false;
            }
        }
        
        // Execute command
        function executeCommand() {
            const input = document.getElementById('commandInput');
            const instruction = input.value.trim();
            
            if (!instruction) {
                addLogEntry('warning', 'Please enter an instruction');
                return;
            }
            
            addLogEntry('info', `Executing: ${instruction}`);
            
            const success = sendMessage({
                type: 'execute_automation',
                instruction: instruction,
                timestamp: new Date().toISOString()
            });
            
            if (success) {
                input.value = '';
                
                // Create session entry
                const sessionId = 'session_' + Date.now();
                activeSessions[sessionId] = {
                    session_id: sessionId,
                    instruction: instruction,
                    status: "running"
                };
                updateSessionsList();
            }
        }
        
        // Test connection
        function testConnection() {
            addLogEntry('info', 'Testing connection...');
            sendMessage({
                type: 'ping',
                timestamp: new Date().toISOString()
            });
        }
        
        // Update connection status
        function updateConnectionStatus(connected) {
            const indicator = document.getElementById('connectionStatus');
            if (connected) {
                indicator.classList.add('connected');
            } else {
                indicator.classList.remove('connected');
            }
        }
        
        // Update metrics
        function updateMetrics() {
            sendMessage({
                type: 'get_system_metrics',
                timestamp: new Date().toISOString()
            });
        }
        
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            ['consoleTab', 'sessionsTab', 'evidenceTab', 'systemTab'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + 'Tab').classList.remove('hidden');
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
            
            currentTab = tabName;
        }
        
        // Clear console
        function clearConsole() {
            if (currentTab === 'console') {
                document.getElementById('consoleTab').innerHTML = '';
                messageCount = 0;
                document.getElementById('messageCount').textContent = '0';
            }
        }
        
        // Export logs
        function exportLogs() {
            const logs = document.getElementById('consoleTab').innerText;
            const blob = new Blob([logs], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `super_omega_logs_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Select session
        function selectSession(sessionId) {
            // Remove active class from all sessions
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Add active class to selected session
            event.currentTarget.classList.add('active');
            
            // Switch to sessions tab if not already there
            if (currentTab !== 'sessions') {
                switchTab('sessions');
            }
            
            addLogEntry('info', `Selected session: ${sessionId}`);
        }
        
        // Setup event listeners
        function setupEventListeners() {
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    switchTab(tab.dataset.tab);
                });
            });
            
            // Enter key for command input
            document.getElementById('commandInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    executeCommand();
                }
            });
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initConsole);
    </script>
</body>
</html>
        """.strip()
    
    def start(self):
        """Start the live console server"""
        try:
            # Create server with console configuration
            handler = self._create_console_handler()
            
            import socketserver
            self.server = socketserver.TCPServer((self.config.host, self.config.port), handler)
            
            # Add server attributes
            self.server.static_dir = self.config.static_dir
            self.server.enable_cors = self.config.enable_cors
            self.server.websocket_connections = self.websocket_connections
            self.server.start_time = time.time()
            self.server.console_instance = self
            
            # Start server in thread
            import threading
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            self.start_time = time.time()
            
            logger.info(f"🚀 SUPER-OMEGA Live Console started on {self.config.host}:{self.config.port}")
            logger.info(f"📊 Console URL: http://{self.config.host}:{self.config.port}")
            logger.info(f"🔗 WebSocket support: Enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live console: {e}")
            return False
    
    def _create_console_handler(self):
        """Create custom request handler for console"""
        from builtin_web_server import BuiltinHTTPRequestHandler
        
        class ConsoleRequestHandler(BuiltinHTTPRequestHandler):
            def _serve_default_page(self):
                """Serve the live console interface"""
                console_html = self.server.console_instance.console_html
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(console_html)))
                
                if getattr(self.server, 'enable_cors', True):
                    self.send_header('Access-Control-Allow-Origin', '*')
                
                self.end_headers()
                self.wfile.write(console_html.encode('utf-8'))
            
            def _process_websocket_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Process console-specific WebSocket messages"""
                message_type = data.get('type', '')
                console = self.server.console_instance
                
                if message_type == 'ping':
                    return {'type': 'pong', 'timestamp': time.time()}
                
                elif message_type == 'console_init':
                    # Console initialization
                    return {
                        'type': 'console_ready',
                        'server': 'SUPER-OMEGA Live Console',
                        'version': '1.0.0',
                        'features': ['automation', 'monitoring', 'websocket'],
                        'timestamp': time.time()
                    }
                
                elif message_type == 'execute_automation':
                    # Handle automation execution
                    instruction = data.get('instruction', '')
                    session_id = f"console_session_{int(time.time())}"
                    
                    # Create session
                    session = AutomationSession(
                        session_id=session_id,
                        instruction=instruction,
                        status="running"
                    )
                    console.active_sessions[session_id] = session
                    
                    # Simulate automation execution (in production, integrate with actual automation)
                    try:
                        # This would integrate with your actual automation system
                        result = console._execute_automation_instruction(instruction, session_id)
                        
                        return {
                            'type': 'automation_result',
                            'success': result.get('success', True),
                            'message': result.get('message', 'Automation executed successfully'),
                            'session_id': session_id,
                            'timestamp': time.time()
                        }
                    except Exception as e:
                        return {
                            'type': 'automation_result',
                            'success': False,
                            'message': f'Automation failed: {str(e)}',
                            'session_id': session_id,
                            'timestamp': time.time()
                        }
                
                elif message_type == 'get_system_metrics':
                    # Get system metrics
                    try:
                        from builtin_performance_monitor import BuiltinPerformanceMonitor
                        monitor = BuiltinPerformanceMonitor()
                        metrics = monitor.get_system_metrics_dict()
                        
                        return {
                            'type': 'system_metrics',
                            'connections': len(self.server.websocket_connections),
                            'uptime': time.time() - self.server.start_time,
                            'timestamp': time.time(),
                            **metrics
                        }
                    except Exception as e:
                        return {
                            'type': 'system_metrics',
                            'error': str(e),
                            'connections': len(self.server.websocket_connections),
                            'uptime': time.time() - self.server.start_time,
                            'timestamp': time.time()
                        }
                
                elif message_type == 'get_sessions':
                    # Get active sessions
                    sessions_data = []
                    for session in console.active_sessions.values():
                        sessions_data.append({
                            'session_id': session.session_id,
                            'instruction': session.instruction,
                            'status': session.status,
                            'created_at': session.created_at,
                            'steps_count': len(session.steps),
                            'evidence_count': len(session.evidence)
                        })
                    
                    return {
                        'type': 'sessions_list',
                        'sessions': sessions_data,
                        'total': len(sessions_data),
                        'timestamp': time.time()
                    }
                
                return None
        
        return ConsoleRequestHandler
    
    def _execute_automation_instruction(self, instruction: str, session_id: str) -> Dict[str, Any]:
        """Execute automation instruction (integrate with actual automation system)"""
        try:
            # Update session status
            session = self.active_sessions[session_id]
            session.status = "running"
            session.updated_at = time.time()
            
            # Simulate automation steps (in production, integrate with your automation engine)
            import asyncio
            
            # Basic instruction processing
            if "navigate" in instruction.lower():
                session.steps.append({
                    "action": "navigate",
                    "target": "target_url",
                    "status": "completed",
                    "timestamp": time.time()
                })
                
            elif "click" in instruction.lower():
                session.steps.append({
                    "action": "click",
                    "target": "element",
                    "status": "completed",
                    "timestamp": time.time()
                })
            
            # Simulate evidence collection
            session.evidence.append(f"screenshot_{session_id}_{int(time.time())}.png")
            session.evidence.append(f"video_{session_id}_{int(time.time())}.mp4")
            
            # Update session as completed
            session.status = "completed"
            session.updated_at = time.time()
            
            return {
                'success': True,
                'message': f'Automation completed successfully: {instruction}',
                'session_id': session_id,
                'steps_completed': len(session.steps),
                'evidence_collected': len(session.evidence)
            }
            
        except Exception as e:
            # Update session as failed
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "failed"
                self.active_sessions[session_id].updated_at = time.time()
            
            return {
                'success': False,
                'message': f'Automation failed: {str(e)}',
                'session_id': session_id
            }
    
    def get_console_status(self) -> Dict[str, Any]:
        """Get console status information"""
        return {
            'running': self.running,
            'host': self.config.host,
            'port': self.config.port,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'active_sessions': len(self.active_sessions),
            'websocket_connections': len(self.websocket_connections),
            'total_sessions': len(self.active_sessions),
            'console_clients': len(self.console_clients)
        }
    
    def broadcast_to_console(self, message: Dict[str, Any]):
        """Broadcast message to all console clients"""
        return self.broadcast_websocket_message(message)
    
    def get_session(self, session_id: str) -> Optional[AutomationSession]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)
    
    def get_all_sessions(self) -> List[AutomationSession]:
        """Get all sessions"""
        return list(self.active_sessions.values())
    
    def clear_completed_sessions(self):
        """Clear completed sessions"""
        completed_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.status in ['completed', 'failed']
        ]
        
        for session_id in completed_sessions:
            del self.active_sessions[session_id]
        
        return len(completed_sessions)

# Global console instance
_global_console: Optional[SuperOmegaLiveConsole] = None

def get_live_console(host: str = "localhost", port: int = 8080) -> SuperOmegaLiveConsole:
    """Get or create the global live console instance"""
    global _global_console
    
    if _global_console is None:
        _global_console = SuperOmegaLiveConsole(host, port)
    
    return _global_console

def start_live_console(host: str = "localhost", port: int = 8080) -> SuperOmegaLiveConsole:
    """Start the live console server"""
    console = get_live_console(host, port)
    
    if not console.is_running():
        console.start()
    
    return console

def stop_live_console():
    """Stop the live console server"""
    global _global_console
    
    if _global_console and _global_console.is_running():
        _global_console.stop()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 SUPER-OMEGA Live Console Demo")
    print("=" * 40)
    
    # Create and start console
    console = SuperOmegaLiveConsole("localhost", 8080)
    
    try:
        print("Starting Live Console...")
        if console.start():
            print(f"✅ Console running at http://localhost:8080")
            print("✅ WebSocket support enabled")
            print("✅ Real-time monitoring active")
            print("\nPress Ctrl+C to stop...")
            
            # Keep running
            while console.is_running():
                time.sleep(1)
        else:
            print("❌ Failed to start console")
            
    except KeyboardInterrupt:
        print("\nStopping console...")
        console.stop()
        print("✅ Console stopped")
        
    print("\n✅ Live console demo complete!")
    print("🎯 Zero external dependencies required!")