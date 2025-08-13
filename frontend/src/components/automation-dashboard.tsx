'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Cpu,
  Memory,
  HardDrive,
  Network,
  Activity,
  Play,
  Pause,
  Stop,
  Settings,
  BarChart3,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Zap,
  Users,
  Globe,
  Database,
  Server
} from 'lucide-react';

interface AutomationMetrics {
  executionTime: number;
  memoryUsage: number;
  cpuUsage: number;
  networkUsage: number;
  diskUsage: number;
  activeAutomations: number;
  totalAutomations: number;
  successRate: number;
  errorRate: number;
}

interface AutomationAgent {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'error' | 'offline';
  currentTask?: string;
  performance: {
    cpu: number;
    memory: number;
    responseTime: number;
  };
  lastActivity: Date;
}

interface AutomationDashboardProps {
  metrics: AutomationMetrics;
  agents: AutomationAgent[];
  onAgentControl: (agentId: string, action: 'start' | 'stop' | 'restart') => void;
  onViewDetails: (agentId: string) => void;
}

export default function AutomationDashboard({
  metrics,
  agents,
  onAgentControl,
  onViewDetails
}: AutomationDashboardProps) {
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        // Refresh metrics every 5 seconds
        console.log('Refreshing metrics...');
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'idle': return 'text-yellow-600 bg-yellow-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'offline': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'idle': return <Clock className="w-4 h-4" />;
      case 'error': return <XCircle className="w-4 h-4" />;
      case 'offline': return <AlertCircle className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Automation Dashboard</h2>
            <p className="text-sm text-gray-500">Real-time monitoring and control</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Time Range:</span>
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value as any)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="autoRefresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="autoRefresh" className="text-sm text-gray-600">
              Auto Refresh
            </label>
          </div>
          
          <button className="p-2 rounded-lg hover:bg-gray-100">
            <Settings className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-90">Active Automations</p>
              <p className="text-2xl font-bold">{metrics.activeAutomations}</p>
            </div>
            <Activity className="w-8 h-8 opacity-80" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 bg-gradient-to-r from-green-500 to-green-600 rounded-lg text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-90">Success Rate</p>
              <p className="text-2xl font-bold">{metrics.successRate.toFixed(1)}%</p>
            </div>
            <CheckCircle className="w-8 h-8 opacity-80" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-4 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-90">CPU Usage</p>
              <p className="text-2xl font-bold">{metrics.cpuUsage.toFixed(1)}%</p>
            </div>
            <Cpu className="w-8 h-8 opacity-80" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="p-4 bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-90">Memory Usage</p>
              <p className="text-2xl font-bold">{metrics.memoryUsage.toFixed(1)}%</p>
            </div>
            <Memory className="w-8 h-8 opacity-80" />
          </div>
        </motion.div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            System Performance
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>CPU Usage</span>
                <span>{metrics.cpuUsage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  className="bg-blue-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.cpuUsage}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Memory Usage</span>
                <span>{metrics.memoryUsage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  className="bg-green-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.memoryUsage}%` }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Network Usage</span>
                <span>{metrics.networkUsage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  className="bg-purple-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.networkUsage}%` }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5" />
            Automation Statistics
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">Total Automations</span>
              <span className="font-semibold">{metrics.totalAutomations}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Success Rate</span>
              <span className="font-semibold text-green-600">{metrics.successRate.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Error Rate</span>
              <span className="font-semibold text-red-600">{metrics.errorRate.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Avg Execution Time</span>
              <span className="font-semibold">{metrics.executionTime.toFixed(1)}s</span>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Status */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Users className="w-5 h-5" />
          Agent Status
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-lg p-4 border border-gray-200 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${
                    agent.status === 'active' ? 'bg-green-500' :
                    agent.status === 'idle' ? 'bg-yellow-500' :
                    agent.status === 'error' ? 'bg-red-500' :
                    'bg-gray-500'
                  }`} />
                  <span className="font-medium">{agent.name}</span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(agent.status)}`}>
                  {agent.status}
                </span>
              </div>
              
              {agent.currentTask && (
                <p className="text-sm text-gray-600 mb-3">
                  Current: {agent.currentTask}
                </p>
              )}
              
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-xs">
                  <span>CPU</span>
                  <span>{agent.performance.cpu.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Memory</span>
                  <span>{agent.performance.memory.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Response</span>
                  <span>{agent.performance.responseTime.toFixed(0)}ms</span>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onViewDetails(agent.id)}
                  className="flex-1 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors"
                >
                  Details
                </button>
                <button
                  onClick={() => onAgentControl(agent.id, 'restart')}
                  className="px-2 py-1 bg-gray-500 text-white text-xs rounded hover:bg-gray-600 transition-colors"
                >
                  Restart
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}