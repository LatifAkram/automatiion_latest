'use client';

import React, { useState, useEffect } from 'react';
import SimpleChatInterface from '../src/components/simple-chat-interface';
import AutomationDashboard from '../src/components/automation-dashboard';

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  automation?: {
    type: string;
    status: 'running' | 'completed' | 'failed';
    progress: number;
    automationId?: string;
    screenshots?: Array<{
      path: string;
      timestamp: string;
      action: string;
    }>;
  };
  sources?: Array<{
    title: string;
    url: string;
    snippet: string;
    domain: string;
    relevance: number;
    source: string;
  }>;
  files?: Array<{
    name: string;
    type: string;
    size: string;
    url: string;
  }>;
  isExpanded?: boolean;
}

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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [activeAutomation, setActiveAutomation] = useState<string | null>(null);
  const [showDashboard, setShowDashboard] = useState(false);
  const [automationMetrics, setAutomationMetrics] = useState<AutomationMetrics>({
    executionTime: 0,
    memoryUsage: 0,
    cpuUsage: 0,
    networkUsage: 0,
    diskUsage: 0,
    activeAutomations: 0,
    totalAutomations: 0,
    successRate: 0,
    errorRate: 0
  });
  const [agents, setAgents] = useState<AutomationAgent[]>([]);

  // Initialize agents
  useEffect(() => {
    setAgents([
      {
        id: 'planner',
        name: 'Planner Agent',
        status: 'active',
        currentTask: 'Workflow planning',
        performance: { cpu: 15, memory: 25, responseTime: 120 },
        lastActivity: new Date()
      },
      {
        id: 'executor',
        name: 'Executor Agent',
        status: 'active',
        currentTask: 'Automation execution',
        performance: { cpu: 45, memory: 60, responseTime: 85 },
        lastActivity: new Date()
      },
      {
        id: 'search',
        name: 'Search Agent',
        status: 'active',
        currentTask: 'Web search',
        performance: { cpu: 20, memory: 30, responseTime: 200 },
        lastActivity: new Date()
      },
      {
        id: 'conversational',
        name: 'Conversational Agent',
        status: 'active',
        currentTask: 'Chat processing',
        performance: { cpu: 10, memory: 15, responseTime: 50 },
        lastActivity: new Date()
      },
      {
        id: 'dom_extractor',
        name: 'DOM Extractor Agent',
        status: 'active',
        currentTask: 'Data extraction',
        performance: { cpu: 25, memory: 35, responseTime: 150 },
        lastActivity: new Date()
      }
    ]);
  }, []);

  const handleSendMessage = async (message: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      // Analyze message for automation intent
      const isAutomationRequest = message.toLowerCase().includes('automate') || 
                                 message.toLowerCase().includes('book') ||
                                 message.toLowerCase().includes('search') ||
                                 message.toLowerCase().includes('extract') ||
                                 message.toLowerCase().includes('fill') ||
                                 message.toLowerCase().includes('monitor');

      // Send message to backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: 'default_session',
          context: {
            domain: 'general',
            user_preferences: {
              automation_type: isAutomationRequest ? 'automation' : 'general',
              complexity: 'medium'
            }
          }
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: data.response || 'I understand your request. Let me help you with that.',
          timestamp: new Date(),
          automation: {
            type: isAutomationRequest ? 'workflow_creation' : 'chat_response',
            status: 'running',
            progress: 0
          }
        };

        setMessages(prev => [...prev, aiMessage]);
        setActiveAutomation(aiMessage.id);

        // If it's an automation request, execute it
        if (isAutomationRequest) {
          await executeAutomation(message, aiMessage.id);
        } else {
          // Simulate chat response completion
          setTimeout(() => {
            setMessages(prev => prev.map(msg => 
              msg.id === aiMessage.id 
                ? { 
                    ...msg, 
                    automation: { ...msg.automation!, status: 'completed', progress: 100 }
                  }
                : msg
            ));
            setActiveAutomation(null);
          }, 2000);
        }

      } else {
        throw new Error('Failed to get response from AI');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        status: 'error'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const executeAutomation = async (message: string, messageId: string) => {
    try {
      // Determine automation type based on message
      let automationType = 'web_automation';
      let url = '';
      let actions = [];

      if (message.toLowerCase().includes('book') && message.toLowerCase().includes('flight')) {
        // Ticket booking automation
        const bookingResponse = await fetch('/automation/ticket-booking', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            from: 'Delhi',
            to: 'Mumbai',
            date: 'Friday',
            time: '6 AM IST',
            passengers: 1,
            budget: '₹8,000'
          })
        });

        if (bookingResponse.ok) {
          const bookingData = await bookingResponse.json();
          updateAutomationProgress(messageId, 100, 'completed', bookingData.result);
          return;
        }
      } else if (message.toLowerCase().includes('search')) {
        // Web search automation
        url = 'https://www.google.com';
        actions = [
          { type: 'navigate' },
          { type: 'wait', time: 2000 },
          { type: 'screenshot' }
        ];
      } else {
        // General web automation
        url = 'https://httpbin.org/forms/post';
        actions = [
          { type: 'navigate' },
          { type: 'wait', time: 2000 },
          { type: 'screenshot' }
        ];
      }

      // Execute automation
      const automationResponse = await fetch('/automation/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: automationType,
          url,
          actions,
          options: { headless: true }
        })
      });

      if (automationResponse.ok) {
        const automationData = await automationResponse.json();
        updateAutomationProgress(messageId, 100, 'completed', automationData.result);
      } else {
        throw new Error('Automation execution failed');
      }

    } catch (error) {
      console.error('Automation error:', error);
      updateAutomationProgress(messageId, 0, 'failed');
    }
  };

  const updateAutomationProgress = (messageId: string, progress: number, status: 'running' | 'completed' | 'failed', result?: any) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { 
            ...msg, 
            automation: { 
              ...msg.automation!, 
              progress, 
              status,
              automationId: result?.automation_id,
              screenshots: result?.screenshots
            }
          }
        : msg
    ));

    if (status === 'completed' || status === 'failed') {
      setActiveAutomation(null);
    }
  };

  const handleAutomationControl = (action: string, messageId: string) => {
    console.log(`Automation control: ${action} for message ${messageId}`);
    
    if (action === 'play') {
      setActiveAutomation(messageId);
    } else if (action === 'pause') {
      setActiveAutomation(null);
    }
  };

  const handleUserInput = (messageId: string, data: any) => {
    console.log(`User input for message ${messageId}:`, data);
  };

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    console.log('Copied to clipboard:', text);
  };

  const handleAgentControl = (agentId: string, action: 'start' | 'stop' | 'restart') => {
    console.log(`Agent control: ${action} for agent ${agentId}`);
    // Update agent status
    setAgents(prev => prev.map(agent => 
      agent.id === agentId 
        ? { ...agent, status: action === 'start' ? 'active' : 'idle' }
        : agent
    ));
  };

  const handleViewDetails = (agentId: string) => {
    console.log(`View details for agent ${agentId}`);
  };

  return (
    <div className="h-screen flex">
      {/* Main Chat Interface */}
      <div className={`flex-1 ${showDashboard ? 'w-2/3' : 'w-full'} transition-all duration-300`}>
        <SimpleChatInterface
          messages={messages}
          isTyping={isTyping}
          activeAutomation={activeAutomation}
          onSendMessage={handleSendMessage}
          onAutomationControl={handleAutomationControl}
          onUserInput={handleUserInput}
          onCopyToClipboard={handleCopyToClipboard}
        />
      </div>

      {/* Automation Dashboard */}
      {showDashboard && (
        <div className="w-1/3 border-l border-gray-200">
          <AutomationDashboard
            metrics={automationMetrics}
            agents={agents}
            onAgentControl={handleAgentControl}
            onViewDetails={handleViewDetails}
          />
        </div>
      )}

      {/* Toggle Dashboard Button */}
      <button
        onClick={() => setShowDashboard(!showDashboard)}
        className="fixed top-4 right-4 z-50 p-3 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors"
        title={showDashboard ? 'Hide Dashboard' : 'Show Dashboard'}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </button>
    </div>
  );
}