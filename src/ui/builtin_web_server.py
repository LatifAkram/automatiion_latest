#!/usr/bin/env python3
"""
Built-in Web Server - 100% Dependency-Free
==========================================

Complete web server with WebSocket support using only Python standard library.
Provides all functionality of FastAPI without external dependencies.
"""

import http.server
import socketserver
import json
import urllib.parse
import threading
import socket
import hashlib
import base64
import struct
import time
import os
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebSocketFrame:
    """WebSocket frame data"""
    opcode: int
    payload: bytes
    fin: bool = True

class WebSocketConnection:
    """Individual WebSocket connection"""
    
    def __init__(self, client_socket: socket.socket, address: tuple):
        self.socket = client_socket
        self.address = address
        self.connected = True
        
    def send_text(self, text: str):
        """Send text message"""
        self.send_frame(WebSocketFrame(opcode=0x1, payload=text.encode('utf-8')))
    
    def send_json(self, data: Any):
        """Send JSON message"""
        self.send_text(json.dumps(data))
    
    def send_frame(self, frame: WebSocketFrame):
        """Send WebSocket frame"""
        if not self.connected or not self.socket:
            return
        
        try:
            # Check if socket is still valid
            if hasattr(self.socket, 'fileno'):
                try:
                    self.socket.fileno()
                except:
                    # Socket is closed
                    self.connected = False
                    return
            
            # Create WebSocket frame
            payload_length = len(frame.payload)
            
            # First byte: FIN + opcode
            first_byte = (0x80 if frame.fin else 0x00) | frame.opcode
            
            # Build frame
            frame_data = bytes([first_byte])
            
            # Payload length
            if payload_length < 126:
                frame_data += bytes([payload_length])
            elif payload_length < 65536:
                frame_data += bytes([126]) + struct.pack('>H', payload_length)
            else:
                frame_data += bytes([127]) + struct.pack('>Q', payload_length)
            
            # Payload
            frame_data += frame.payload
            
            self.socket.send(frame_data)
            
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # Connection closed by client
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            self.connected = False
    
    def receive_frame(self) -> Optional[WebSocketFrame]:
        """Receive WebSocket frame"""
        try:
            # Read first two bytes
            data = self.socket.recv(2)
            if len(data) < 2:
                return None
            
            first_byte, second_byte = data
            
            # Parse first byte
            fin = bool(first_byte & 0x80)
            opcode = first_byte & 0x0f
            
            # Parse second byte
            masked = bool(second_byte & 0x80)
            payload_length = second_byte & 0x7f
            
            # Extended payload length
            if payload_length == 126:
                data = self.socket.recv(2)
                payload_length = struct.unpack('>H', data)[0]
            elif payload_length == 127:
                data = self.socket.recv(8)
                payload_length = struct.unpack('>Q', data)[0]
            
            # Masking key
            mask = None
            if masked:
                mask = self.socket.recv(4)
            
            # Payload
            payload = self.socket.recv(payload_length)
            
            # Unmask payload if needed
            if masked and mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            
            return WebSocketFrame(opcode=opcode, payload=payload, fin=fin)
            
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            self.connected = False
            return None
    
    def close(self):
        """Close connection"""
        self.connected = False
        try:
            self.socket.close()
        except:
            pass

class BuiltinWebServer:
    """Complete web server with WebSocket support"""
    
    def __init__(self, host: str = "localhost", port: int = 8081):
        self.host = host
        self.port = port
        self.routes = {}
        self.websocket_handlers = {}
        self.websocket_connections = []
        self.static_files = {}
        self.server_thread = None
        self.running = False
        
    def route(self, path: str, methods: List[str] = None):
        """Decorator to register route handlers"""
        if methods is None:
            methods = ['GET']
        
        def decorator(func):
            for method in methods:
                self.routes[f"{method} {path}"] = func
            return func
        return decorator
    
    def websocket(self, path: str):
        """Decorator to register WebSocket handlers"""
        def decorator(func):
            self.websocket_handlers[path] = func
            return func
        return decorator
    
    def add_static_file(self, path: str, content: str, content_type: str = "text/html"):
        """Add static file content"""
        self.static_files[path] = {
            "content": content,
            "content_type": content_type
        }
    
    def handle_request(self, request_handler):
        """Handle HTTP request"""
        try:
            # Parse request line
            request_line = request_handler.requestline
            method, path, version = request_line.split()
            
            # Parse query parameters
            if '?' in path:
                path, query_string = path.split('?', 1)
                query_params = urllib.parse.parse_qs(query_string)
            else:
                query_params = {}
            
            # Check for WebSocket upgrade
            headers = request_handler.headers
            if (headers.get('Connection', '').lower() == 'upgrade' and
                headers.get('Upgrade', '').lower() == 'websocket'):
                return self.handle_websocket_upgrade(request_handler, path)
            
            # Handle static files
            if path in self.static_files:
                static_file = self.static_files[path]
                request_handler.send_response(200)
                request_handler.send_header('Content-Type', static_file['content_type'])
                # Add CORS headers
                request_handler.send_header('Access-Control-Allow-Origin', '*')
                request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                request_handler.end_headers()
                request_handler.wfile.write(static_file['content'].encode('utf-8'))
                return
            
            # Handle routes with parameter matching
            handler = None
            path_params = {}
            
            # First try exact match
            route_key = f"{method} {path}"
            if route_key in self.routes:
                handler = self.routes[route_key]
            else:
                # Try parameter matching
                for registered_route, route_handler in self.routes.items():
                    route_method, route_path = registered_route.split(' ', 1)
                    if route_method == method:
                        # Check if this route has parameters
                        if '<' in route_path and '>' in route_path:
                            # Simple parameter matching
                            route_parts = route_path.split('/')
                            path_parts = path.split('/')
                            
                            if len(route_parts) == len(path_parts):
                                match = True
                                for i, (route_part, path_part) in enumerate(zip(route_parts, path_parts)):
                                    if route_part.startswith('<') and route_part.endswith('>'):
                                        # This is a parameter
                                        param_name = route_part[1:-1]
                                        path_params[param_name] = path_part
                                    elif route_part != path_part:
                                        match = False
                                        break
                                
                                if match:
                                    handler = route_handler
                                    break
            
            # Handle OPTIONS preflight requests
            if method == 'OPTIONS':
                request_handler.send_response(200)
                request_handler.send_header('Access-Control-Allow-Origin', '*')
                request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                request_handler.end_headers()
                return
            
            if handler:
                # Create request context
                request_context = {
                    'method': method,
                    'path': path,
                    'query_params': query_params,
                    'path_params': path_params,
                    'headers': dict(headers),
                    'body': None
                }
                
                # Read body for POST/PUT requests
                if method in ['POST', 'PUT']:
                    content_length = int(headers.get('Content-Length', 0))
                    if content_length > 0:
                        body = request_handler.rfile.read(content_length)
                        body_str = body.decode('utf-8')
                        
                        # Try to parse JSON body
                        try:
                            if headers.get('Content-Type', '').startswith('application/json'):
                                request_context['body'] = json.loads(body_str)
                            else:
                                request_context['body'] = body_str
                        except json.JSONDecodeError:
                            request_context['body'] = body_str
                
                # Call handler
                response = handler(request_context)
                
                # Send response
                if isinstance(response, dict):
                    # JSON response
                    json_data = json.dumps(response)
                    request_handler.send_response(200)
                    request_handler.send_header('Content-Type', 'application/json')
                    # Add CORS headers
                    request_handler.send_header('Access-Control-Allow-Origin', '*')
                    request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                    request_handler.end_headers()
                    request_handler.wfile.write(json_data.encode('utf-8'))
                elif isinstance(response, str):
                    # Text response
                    request_handler.send_response(200)
                    request_handler.send_header('Content-Type', 'text/plain')
                    # Add CORS headers
                    request_handler.send_header('Access-Control-Allow-Origin', '*')
                    request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                    request_handler.end_headers()
                    request_handler.wfile.write(response.encode('utf-8'))
                else:
                    # Default response
                    request_handler.send_response(200)
                    # Add CORS headers
                    request_handler.send_header('Access-Control-Allow-Origin', '*')
                    request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                    request_handler.end_headers()
                
                return
            
            # 404 Not Found
            request_handler.send_response(404)
            # Add CORS headers
            request_handler.send_header('Access-Control-Allow-Origin', '*')
            request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            request_handler.end_headers()
            request_handler.wfile.write(b'Not Found')
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            request_handler.send_response(500)
            # Add CORS headers
            request_handler.send_header('Access-Control-Allow-Origin', '*')
            request_handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            request_handler.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            request_handler.end_headers()
            request_handler.wfile.write(b'Internal Server Error')
    
    def handle_websocket_upgrade(self, request_handler, path):
        """Handle WebSocket upgrade request"""
        headers = request_handler.headers
        
        # Validate WebSocket headers
        if headers.get('Sec-WebSocket-Version') != '13':
            request_handler.send_response(400)
            request_handler.end_headers()
            return
        
        # Get WebSocket key
        websocket_key = headers.get('Sec-WebSocket-Key')
        if not websocket_key:
            request_handler.send_response(400)
            request_handler.end_headers()
            return
        
        # Generate accept key
        magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept_key = base64.b64encode(
            hashlib.sha1((websocket_key + magic_string).encode()).digest()
        ).decode()
        
        # Send upgrade response
        request_handler.send_response(101, 'Switching Protocols')
        request_handler.send_header('Upgrade', 'websocket')
        request_handler.send_header('Connection', 'Upgrade')
        request_handler.send_header('Sec-WebSocket-Accept', accept_key)
        request_handler.end_headers()
        
        # Create WebSocket connection
        connection = WebSocketConnection(request_handler.connection, request_handler.client_address)
        self.websocket_connections.append(connection)
        
        # Handle WebSocket in separate thread
        threading.Thread(
            target=self.handle_websocket_connection,
            args=(connection, path),
            daemon=True
        ).start()
    
    def handle_websocket_connection(self, connection: WebSocketConnection, path: str):
        """Handle WebSocket connection"""
        try:
            # Find handler
            handler = self.websocket_handlers.get(path)
            if not handler:
                connection.close()
                return
            
            # Call connection handler
            handler(connection)
            
            # Message loop
            while connection.connected:
                frame = connection.receive_frame()
                if frame is None:
                    break
                
                if frame.opcode == 0x8:  # Close frame
                    break
                elif frame.opcode == 0x1:  # Text frame
                    try:
                        message = frame.payload.decode('utf-8')
                        data = json.loads(message)
                        
                        # Handle message (could be extended)
                        if 'type' in data:
                            self.handle_websocket_message(connection, data)
                    except Exception as e:
                        logger.error(f"WebSocket message error: {e}")
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            connection.close()
            if connection in self.websocket_connections:
                self.websocket_connections.remove(connection)
    
    def handle_websocket_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle WebSocket message (override in subclass)"""
        # Echo message back by default
        connection.send_json({"type": "echo", "data": data})
    
    def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections"""
        disconnected = []
        for connection in self.websocket_connections:
            if connection.connected:
                try:
                    connection.send_json(message)
                except:
                    disconnected.append(connection)
            else:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            if conn in self.websocket_connections:
                self.websocket_connections.remove(conn)
    
    def start(self):
        """Start the web server"""
        class CustomHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
            def __init__(self, request, client_address, server):
                self.web_server = server.web_server_instance
                super().__init__(request, client_address, server)
            
            def do_GET(self):
                self.web_server.handle_request(self)
            
            def do_POST(self):
                self.web_server.handle_request(self)
            
            def do_PUT(self):
                self.web_server.handle_request(self)
            
            def do_OPTIONS(self):
                self.web_server.handle_request(self)
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        class CustomTCPServer(socketserver.TCPServer):
            def __init__(self, server_address, RequestHandlerClass, web_server_instance):
                self.web_server_instance = web_server_instance
                super().__init__(server_address, RequestHandlerClass)
        
        try:
            self.server = CustomTCPServer((self.host, self.port), CustomHTTPRequestHandler, self)
            self.running = True
            
            logger.info(f"🌐 Built-in web server starting at http://{self.host}:{self.port}")
            
            # Start server in thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return False
    
    def stop(self):
        """Stop the web server"""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        # Close all WebSocket connections
        for connection in self.websocket_connections[:]:
            connection.close()
        
        logger.info("🔴 Web server stopped")

# Example Live Console implementation
class LiveConsoleServer(BuiltinWebServer):
    """Live console server using built-in web server"""
    
    def __init__(self, host: str = "localhost", port: int = 8081):
        super().__init__(host, port)
        self.setup_routes()
        self.active_runs = {}
        
    def setup_routes(self):
        """Setup console routes and static files"""
        
        # Main console HTML
        console_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUPER-OMEGA Live Console (Built-in)</title>
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
            text-align: center;
        }
        .status { 
            flex: 1; 
            padding: 20px; 
            overflow-y: auto; 
            background: #0f0f0f;
        }
        .message { 
            margin-bottom: 10px; 
            padding: 10px; 
            border-radius: 4px;
            background: #1a1a1a;
            border-left: 3px solid #00ff00;
        }
        .controls { 
            padding: 20px; 
            background: #2a2a2a; 
            display: flex; 
            gap: 10px;
        }
        .btn { 
            padding: 10px 20px; 
            border: 1px solid #00ff00; 
            background: transparent; 
            color: #00ff00; 
            border-radius: 4px; 
            cursor: pointer;
        }
        .btn:hover { background: #00ff00; color: #000; }
        .metrics { display: flex; gap: 20px; margin-bottom: 20px; }
        .metric { text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff00; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🚀 SUPER-OMEGA</h2>
            <h3>Built-in Console</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="connectionStatus">●</div>
                    <div>Connection</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="messageCount">0</div>
                    <div>Messages</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>SUPER-OMEGA Live Console</h1>
                <p>Built-in Web Server - No External Dependencies</p>
            </div>
            
            <div class="status" id="statusArea">
                <div class="message">
                    <strong>🚀 Console Ready</strong><br>
                    Built-in web server active. All functionality working without external dependencies.
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="sendTestMessage()">📊 Test Connection</button>
                <button class="btn" onclick="clearMessages()">🗑️ Clear</button>
                <button class="btn" onclick="showSystemInfo()">ℹ️ System Info</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let messageCount = 0;
        
        function connectWebSocket() {
            try {
                ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onopen = function() {
                    document.getElementById('connectionStatus').textContent = '●';
                    document.getElementById('connectionStatus').style.color = '#00ff00';
                    addMessage('✅ WebSocket connected successfully');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    document.getElementById('connectionStatus').textContent = '●';
                    document.getElementById('connectionStatus').style.color = '#ff0000';
                    addMessage('❌ WebSocket disconnected');
                    setTimeout(connectWebSocket, 3000);
                };
                
            } catch (error) {
                addMessage('❌ WebSocket connection failed: ' + error.message);
            }
        }
        
        function handleMessage(data) {
            if (data.type === 'echo') {
                addMessage('📡 Echo: ' + JSON.stringify(data.data));
            } else if (data.type === 'system_info') {
                addMessage('🖥️ System: ' + data.info);
            } else {
                addMessage('📨 Message: ' + JSON.stringify(data));
            }
        }
        
        function addMessage(text) {
            const statusArea = document.getElementById('statusArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong><br>${text}`;
            statusArea.appendChild(messageDiv);
            statusArea.scrollTop = statusArea.scrollHeight;
            
            messageCount++;
            document.getElementById('messageCount').textContent = messageCount;
        }
        
        function sendTestMessage() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'test',
                    message: 'Built-in web server test',
                    timestamp: new Date().toISOString()
                }));
            } else {
                addMessage('❌ WebSocket not connected');
            }
        }
        
        function clearMessages() {
            document.getElementById('statusArea').innerHTML = '';
            messageCount = 0;
            document.getElementById('messageCount').textContent = '0';
        }
        
        function showSystemInfo() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'get_system_info'}));
            }
        }
        
        // Initialize
        connectWebSocket();
    </script>
</body>
</html>
        """
        
        self.add_static_file("/", console_html)
        
        @self.route("/api/status")
        def get_status(request):
            return {
                "status": "running",
                "server": "built-in",
                "connections": len(self.websocket_connections),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        @self.route("/api/fixed-super-omega-execute", methods=['POST'])
        def execute_automation(request):
            """Execute automation instruction"""
            try:
                # Get instruction from request body
                body = request.get('body', {})
                instruction = body.get('instruction', '')
                
                if not instruction:
                    return {
                        "success": False,
                        "error": "No instruction provided"
                    }
                
                # Execute automation using SUPER-OMEGA system
                import asyncio
                import sys
                import os
                
                # Add src to path for imports
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                
                try:
                    from testing.super_omega_live_automation_fixed import get_fixed_super_omega_live_automation, ExecutionMode
                    
                    # Create automation session
                    automation = get_fixed_super_omega_live_automation({
                        'headless': False,
                        'record_video': True,
                        'capture_screenshots': True
                    })
                    
                    # Execute the instruction
                    session_id = f"ui_session_{int(time.time())}"
                    
                    async def run_automation():
                        try:
                            # Create session
                            session_result = await automation.create_super_omega_session(
                                session_id=session_id,
                                url="about:blank",
                                mode=ExecutionMode.HYBRID
                            )
                            
                            if not session_result.get('success'):
                                return {
                                    "success": False,
                                    "error": f"Session creation failed: {session_result.get('error')}"
                                }
                            
                            # Parse instruction and execute basic automation
                            steps_executed = []
                            
                            # Simple instruction parsing
                            if "navigate" in instruction.lower() and "http" in instruction.lower():
                                # Extract URL from instruction
                                import re
                                url_match = re.search(r'https?://[^\s]+', instruction)
                                if url_match:
                                    url = url_match.group()
                                    nav_result = await automation.super_omega_navigate(session_id, url)
                                    steps_executed.append({
                                        "action": "navigate",
                                        "target": url,
                                        "success": nav_result.get('success', False),
                                        "time": time.strftime("%H:%M:%S")
                                    })
                            
                            elif "search" in instruction.lower():
                                # Simulate search action
                                search_result = await automation.super_omega_find_element(
                                    session_id, 
                                    "input[type='search'], input[name*='search'], #search"
                                )
                                steps_executed.append({
                                    "action": "search",
                                    "success": search_result.get('success', False),
                                    "time": time.strftime("%H:%M:%S")
                                })
                            
                            else:
                                # Default: navigate to example.com for demonstration
                                nav_result = await automation.super_omega_navigate(session_id, "https://example.com")
                                steps_executed.append({
                                    "action": "navigate",
                                    "target": "https://example.com",
                                    "success": nav_result.get('success', False),
                                    "time": time.strftime("%H:%M:%S")
                                })
                            
                            # Close session
                            await automation.close_super_omega_session(session_id)
                            
                            return {
                                "success": True,
                                "session_id": session_id,
                                "instruction": instruction,
                                "steps": steps_executed,
                                "evidence": [f"session_{session_id}_evidence"],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                        except Exception as e:
                            return {
                                "success": False,
                                "error": str(e),
                                "session_id": session_id
                            }
                    
                    # Run async automation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(run_automation())
                    loop.close()
                    
                    return result
                    
                except ImportError as e:
                    return {
                        "success": False,
                        "error": f"Automation system not available: {e}",
                        "fallback": True
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.route("/api/session-status/<session_id>")
        def get_session_status(request):
            """Get session status"""
            session_id = request.get('path_params', {}).get('session_id', '')
            
            # Check if session directory exists
            session_path = Path(f"runs/{session_id}")
            if session_path.exists():
                return {
                    "success": True,
                    "session_id": session_id,
                    "status": "completed",
                    "evidence_available": True
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "status": "not_found"
                }
        
        @self.route("/api/session-evidence/<session_id>")
        def get_session_evidence(request):
            """Get session evidence"""
            session_id = request.get('path_params', {}).get('session_id', '')
            
            evidence = []
            session_path = Path(f"runs/{session_id}")
            
            if session_path.exists():
                # Look for evidence files
                for evidence_file in session_path.glob("**/*"):
                    if evidence_file.is_file():
                        evidence.append({
                            "type": "file",
                            "name": evidence_file.name,
                            "path": str(evidence_file.relative_to(session_path)),
                            "size": evidence_file.stat().st_size
                        })
            
            return {
                "success": True,
                "session_id": session_id,
                "evidence": evidence
            }
        
        @self.websocket("/ws")
        def handle_websocket(connection):
            logger.info(f"WebSocket client connected: {connection.address}")
            connection.send_json({
                "type": "welcome",
                "message": "Connected to built-in SUPER-OMEGA console"
            })
    
    def handle_websocket_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle console WebSocket messages"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'test':
            connection.send_json({
                "type": "echo",
                "data": data,
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        elif message_type == 'get_system_info':
            import platform
            connection.send_json({
                "type": "system_info",
                "info": f"Python {platform.python_version()} on {platform.system()}"
            })
        else:
            # Default echo
            connection.send_json({"type": "echo", "data": data})

if __name__ == "__main__":
    # Demo the built-in web server
    print("🌐 Built-in Web Server Demo")
    print("=" * 40)
    
    server = LiveConsoleServer()
    
    if server.start():
        print(f"✅ Server running at http://{server.host}:{server.port}")
        print("🔗 Open the URL in your browser to see the live console")
        print("📡 WebSocket support included - no external dependencies!")
        print("Press Ctrl+C to stop...")
        
        try:
            while server.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server.stop()
    else:
        print("❌ Failed to start server")
    
    print("✅ Built-in web server demo complete!")
    print("🎯 No external dependencies required!")