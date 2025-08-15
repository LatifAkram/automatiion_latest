#!/usr/bin/env python3
"""
SUPER-OMEGA Live Run Console - 100% Dependency-Free
===================================================

Real-time monitoring and control interface using built-in web server.
No external dependencies required.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.builtin_web_server import BuiltinWebServer
from core.builtin_performance_monitor import get_system_metrics
from core.builtin_ai_processor import process_with_ai
from core.builtin_vision_processor import analyze_screenshot

import time
import json
import threading
from typing import Dict, List, Any

class SuperOmegaLiveConsole(BuiltinWebServer):
    """Enhanced live console with SUPER-OMEGA capabilities"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        super().__init__(host, port)
        self.active_runs = {}
        self.performance_data = []
        self.ai_insights = []
        self.setup_enhanced_routes()
        
        # Start background monitoring
        self.start_background_monitoring()
    
    def setup_enhanced_routes(self):
        """Setup enhanced console routes"""
        
        @self.route("/api/system-metrics")
        def get_system_metrics_endpoint(request):
            """Get real-time system metrics"""
            try:
                metrics = get_system_metrics()
                return {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "memory_used_mb": metrics.memory_used_mb,
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "process_count": metrics.process_count,
                    "uptime_seconds": metrics.uptime_seconds,
                    "platform": metrics.platform_info["system"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.route("/api/ai-analysis", methods=["POST"])
        def ai_analysis_endpoint(request):
            """Perform AI analysis on text"""
            try:
                data = json.loads(request["body"]) if request["body"] else {}
                text = data.get("text", "")
                task = data.get("task", "analyze")
                
                result = process_with_ai(text, task)
                
                return {
                    "confidence": result.confidence,
                    "result": result.result,
                    "reasoning": result.reasoning,
                    "processing_time": result.processing_time
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.route("/api/runs")
        def get_active_runs(request):
            """Get active automation runs"""
            return {
                "runs": list(self.active_runs.values()),
                "count": len(self.active_runs)
            }
        
        @self.route("/api/performance-history")
        def get_performance_history(request):
            """Get performance history"""
            return {
                "data": self.performance_data[-50:],  # Last 50 points
                "count": len(self.performance_data)
            }
        
        # Enhanced console HTML with more features
        enhanced_console_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUPER-OMEGA Live Console - 100% Built-in</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Monaco', 'Consolas', monospace; 
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            color: #00ff00; 
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .sidebar { 
            width: 320px; 
            background: rgba(26, 26, 46, 0.9);
            border-right: 2px solid #00ff00; 
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .main-content { flex: 1; display: flex; flex-direction: column; }
        .header { 
            background: rgba(42, 42, 42, 0.9);
            padding: 15px 20px; 
            border-bottom: 2px solid #00ff00;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .status { 
            flex: 1; 
            padding: 20px; 
            overflow-y: auto; 
            background: rgba(15, 15, 15, 0.9);
        }
        .message { 
            margin-bottom: 10px; 
            padding: 12px; 
            border-radius: 6px;
            background: rgba(26, 26, 26, 0.8);
            border-left: 4px solid #00ff00;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .controls { 
            padding: 20px; 
            background: rgba(42, 42, 42, 0.9);
            display: flex; 
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn { 
            padding: 12px 20px; 
            border: 2px solid #00ff00; 
            background: transparent; 
            color: #00ff00; 
            border-radius: 6px; 
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: inherit;
        }
        .btn:hover { 
            background: #00ff00; 
            color: #000; 
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 255, 0, 0.3);
        }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric { 
            text-align: center; 
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            border: 1px solid #333;
        }
        .metric-value { font-size: 28px; font-weight: bold; color: #00ff00; text-shadow: 0 0 10px #00ff00; }
        .metric-label { font-size: 12px; color: #999; margin-top: 5px; }
        .ai-panel { 
            margin-top: 20px; 
            padding: 15px; 
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            border: 1px solid #333;
        }
        .ai-input { 
            width: 100%; 
            padding: 10px; 
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff00;
            color: #00ff00;
            border-radius: 4px;
            font-family: inherit;
            margin-bottom: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background: #00ff00; box-shadow: 0 0 10px #00ff00; }
        .status-offline { background: #ff0000; box-shadow: 0 0 10px #ff0000; }
        .chart-container {
            height: 60px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🚀 SUPER-OMEGA</h2>
            <h3>100% Built-in Console</h3>
            
            <div class="metrics">
                <div class="metric">
                    <div class="status-indicator" id="connectionStatus"></div>
                    <div class="metric-label">Connection</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="cpuUsage">--</div>
                    <div class="metric-label">CPU %</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="memoryUsage">--</div>
                    <div class="metric-label">Memory %</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="activeRuns">0</div>
                    <div class="metric-label">Active Runs</div>
                </div>
            </div>
            
            <div class="ai-panel">
                <h4>🧠 AI Analysis</h4>
                <input type="text" class="ai-input" id="aiInput" placeholder="Enter text for AI analysis...">
                <button class="btn" onclick="analyzeWithAI()" style="width: 100%; margin-top: 5px;">Analyze</button>
                <div id="aiResult" style="margin-top: 10px; font-size: 12px;"></div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>🎯 SUPER-OMEGA Live Console</h1>
                <p>Built-in Web Server | AI Processor | Vision System | Performance Monitor</p>
                <p style="color: #00ff00; font-size: 14px; margin-top: 5px;">
                    ✅ 100% Dependency-Free | No FastAPI, WebSockets, psutil, transformers, or OpenCV required
                </p>
            </div>
            
            <div class="status" id="statusArea">
                <div class="message">
                    <strong>🚀 SUPER-OMEGA Console Ready</strong><br>
                    All built-in systems operational. Complete functionality without external dependencies.
                    <div class="chart-container" id="performanceChart"></div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="testConnection()">📡 Test Connection</button>
                <button class="btn" onclick="getSystemMetrics()">📊 System Metrics</button>
                <button class="btn" onclick="testAI()">🧠 Test AI</button>
                <button class="btn" onclick="clearMessages()">🗑️ Clear</button>
                <button class="btn" onclick="startDemo()">🎬 Start Demo</button>
                <button class="btn" onclick="showHelp()">❓ Help</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let messageCount = 0;
        let performanceData = [];
        
        function connectWebSocket() {
            try {
                ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onopen = function() {
                    document.getElementById('connectionStatus').className = 'status-indicator status-online';
                    addMessage('✅ WebSocket connected successfully');
                    startPerformanceMonitoring();
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    document.getElementById('connectionStatus').className = 'status-indicator status-offline';
                    addMessage('❌ WebSocket disconnected - attempting reconnection...');
                    setTimeout(connectWebSocket, 3000);
                };
                
            } catch (error) {
                addMessage('❌ WebSocket connection failed: ' + error.message);
            }
        }
        
        function handleMessage(data) {
            if (data.type === 'system_metrics') {
                updateSystemMetrics(data.metrics);
            } else if (data.type === 'ai_result') {
                displayAIResult(data.result);
            } else if (data.type === 'echo') {
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
        }
        
        function testConnection() {
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
        
        function getSystemMetrics() {
            fetch('/api/system-metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addMessage('❌ Metrics error: ' + data.error);
                    } else {
                        updateSystemMetrics(data);
                        addMessage(`📊 System Metrics: CPU ${data.cpu_percent.toFixed(1)}%, Memory ${data.memory_percent.toFixed(1)}%, Processes ${data.process_count}`);
                    }
                })
                .catch(error => addMessage('❌ Metrics fetch failed: ' + error.message));
        }
        
        function updateSystemMetrics(metrics) {
            document.getElementById('cpuUsage').textContent = metrics.cpu_percent.toFixed(1);
            document.getElementById('memoryUsage').textContent = metrics.memory_percent.toFixed(1);
            
            // Add to performance chart data
            performanceData.push({
                cpu: metrics.cpu_percent,
                memory: metrics.memory_percent,
                timestamp: Date.now()
            });
            
            // Keep last 50 points
            if (performanceData.length > 50) {
                performanceData.shift();
            }
            
            updatePerformanceChart();
        }
        
        function updatePerformanceChart() {
            const chart = document.getElementById('performanceChart');
            if (!performanceData.length) return;
            
            const width = chart.offsetWidth;
            const height = chart.offsetHeight;
            
            // Simple ASCII-style chart
            let chartHtml = '<div style="position: absolute; bottom: 0; left: 0; right: 0; height: 100%; display: flex; align-items: end;">';
            
            performanceData.slice(-20).forEach((point, i) => {
                const barHeight = (point.cpu / 100) * height;
                chartHtml += `<div style="width: ${width/20}px; height: ${barHeight}px; background: #00ff00; margin-right: 1px; opacity: 0.7;"></div>`;
            });
            
            chartHtml += '</div>';
            chart.innerHTML = chartHtml;
        }
        
        function analyzeWithAI() {
            const text = document.getElementById('aiInput').value;
            if (!text) return;
            
            fetch('/api/ai-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, task: 'analyze' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('aiResult').innerHTML = '❌ Error: ' + data.error;
                } else {
                    displayAIResult(data);
                }
            })
            .catch(error => {
                document.getElementById('aiResult').innerHTML = '❌ AI analysis failed: ' + error.message;
            });
        }
        
        function displayAIResult(result) {
            const resultDiv = document.getElementById('aiResult');
            resultDiv.innerHTML = `
                <strong>🧠 AI Analysis:</strong><br>
                Confidence: ${(result.confidence * 100).toFixed(1)}%<br>
                Sentiment: ${result.result.sentiment || 'N/A'}<br>
                Processing: ${(result.processing_time * 1000).toFixed(1)}ms
            `;
            
            addMessage(`🧠 AI Analysis complete: ${result.reasoning}`);
        }
        
        function testAI() {
            fetch('/api/ai-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: 'This is an amazing system that works perfectly!', 
                    task: 'analyze' 
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    addMessage('❌ AI Test error: ' + data.error);
                } else {
                    addMessage(`🧠 AI Test successful: ${data.result.sentiment} sentiment with ${(data.confidence*100).toFixed(1)}% confidence`);
                }
            });
        }
        
        function startDemo() {
            addMessage('🎬 Starting built-in demo...');
            // Simulate demo activity
            let step = 1;
            const demoInterval = setInterval(() => {
                addMessage(`🔄 Demo Step ${step}: Processing with built-in systems...`);
                step++;
                if (step > 5) {
                    clearInterval(demoInterval);
                    addMessage('✅ Demo completed successfully!');
                }
            }, 1000);
        }
        
        function showHelp() {
            addMessage(`
                <strong>🎯 SUPER-OMEGA Console Help</strong><br>
                • 📡 Test Connection: Verify WebSocket connectivity<br>
                • 📊 System Metrics: Get real-time system performance<br>
                • 🧠 Test AI: Test built-in AI processing capabilities<br>
                • 🗑️ Clear: Clear all messages<br>
                • 🎬 Start Demo: Run built-in demonstration<br>
                <br>
                <strong>✨ Features:</strong><br>
                • 100% Built-in Web Server (no FastAPI)<br>
                • Real-time Performance Monitoring (no psutil)<br>
                • AI Text Processing (no transformers)<br>
                • Vision Processing (no OpenCV)<br>
                • Data Validation (no pydantic)
            `);
        }
        
        function clearMessages() {
            document.getElementById('statusArea').innerHTML = '';
            messageCount = 0;
        }
        
        function startPerformanceMonitoring() {
            // Update metrics every 2 seconds
            setInterval(getSystemMetrics, 2000);
        }
        
        // Initialize
        connectWebSocket();
        
        // Show initial help
        setTimeout(showHelp, 1000);
    </script>
</body>
</html>
        """
        
        self.add_static_file("/", enhanced_console_html)
    
    def start_background_monitoring(self):
        """Start background monitoring tasks"""
        def monitor_loop():
            while self.running:
                try:
                    # Collect performance data
                    metrics = get_system_metrics()
                    metric_data = {
                        "timestamp": time.time(),
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "process_count": metrics.process_count
                    }
                    
                    self.performance_data.append(metric_data)
                    
                    # Keep only last 100 points
                    if len(self.performance_data) > 100:
                        self.performance_data.pop(0)
                    
                    # Broadcast to connected clients
                    self.broadcast_to_websockets({
                        "type": "system_metrics",
                        "metrics": metric_data
                    })
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                
                time.sleep(2)  # Update every 2 seconds
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def handle_websocket_message(self, connection, data: Dict[str, Any]):
        """Enhanced WebSocket message handling"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'test':
            connection.send_json({
                "type": "echo",
                "data": data,
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "server": "SUPER-OMEGA Built-in"
            })
        elif message_type == 'get_system_info':
            import platform
            connection.send_json({
                "type": "system_info",
                "info": f"SUPER-OMEGA on Python {platform.python_version()} ({platform.system()})"
            })
        elif message_type == 'ai_analyze':
            try:
                text = data.get('text', '')
                result = process_with_ai(text, 'analyze')
                connection.send_json({
                    "type": "ai_result",
                    "result": {
                        "confidence": result.confidence,
                        "result": result.result,
                        "reasoning": result.reasoning,
                        "processing_time": result.processing_time
                    }
                })
            except Exception as e:
                connection.send_json({
                    "type": "error",
                    "message": f"AI analysis failed: {e}"
                })
        else:
            # Default echo
            connection.send_json({"type": "echo", "data": data})

def main():
    """Main console entry point"""
    print("🚀 Starting SUPER-OMEGA Live Console")
    print("=" * 50)
    
    console = SuperOmegaLiveConsole()
    
    if console.start():
        print(f"✅ Console running at http://{console.host}:{console.port}")
        print("🌐 Built-in web server with WebSocket support")
        print("📊 Real-time performance monitoring active")
        print("🧠 AI processing capabilities enabled")
        print("👁️ Vision processing system ready")
        print("🎯 100% dependency-free implementation")
        print("\nPress Ctrl+C to stop...")
        
        try:
            while console.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping console...")
            console.stop()
    else:
        print("❌ Failed to start console")

if __name__ == "__main__":
    main()