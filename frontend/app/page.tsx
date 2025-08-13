'use client';

import React, { useState, useEffect, useRef } from 'react';
import SimpleChatInterface from '../src/components/simple-chat-interface';
import AutomationDashboard from '../src/components/automation-dashboard';

// Backend configuration
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// Theme configuration
type Theme = 'light' | 'dark' | 'auto';

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  automation?: {
    type: string;
    status: 'running' | 'completed' | 'failed' | 'paused' | 'handoff_required';
    progress: number;
    automationId?: string;
    screenshots?: Array<{
      path: string;
      timestamp: string;
      action: string;
    }>;
    handoffReason?: string;
    takeoverData?: any;
  };
  sources?: Array<{
    title: string;
    url: string;
    snippet: string;
    domain: string;
    relevance: number;
    source: string;
    confidence?: number;
    content_type?: string;
    timestamp?: string;
  }>;
  files?: Array<{
    name: string;
    type: string;
    size: string;
    url: string;
  }>;
  isExpanded?: boolean;
  chatId?: string;
  agentType?: 'planner' | 'executor' | 'conversational' | 'search' | 'dom_analysis';
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  lastActivity: Date;
  isActive: boolean;
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
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>('default');
  const [isTyping, setIsTyping] = useState(false);
  const [activeAutomation, setActiveAutomation] = useState<string | null>(null);
  const [showDashboard, setShowDashboard] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [handoffRequests, setHandoffRequests] = useState<any[]>([]);
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
  const [liveAutomationStream, setLiveAutomationStream] = useState<any>(null);
  const automationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Enhanced agentic features
  const [theme, setTheme] = useState<Theme>('auto');
  const [currentAgent, setCurrentAgent] = useState<string>('conversational');
  const [agentStatus, setAgentStatus] = useState<'thinking' | 'processing' | 'idle'>('idle');
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  const [agentAnimations, setAgentAnimations] = useState<{[key: string]: boolean}>({});
  const [showFloatingAgents, setShowFloatingAgents] = useState(false);
  const [agentThoughts, setAgentThoughts] = useState<string[]>([]);
  const [showThoughtBubble, setShowThoughtBubble] = useState(false);

  // Initialize default chat session
  useEffect(() => {
    const defaultSession: ChatSession = {
      id: 'default',
      title: 'Main Chat',
      messages: [],
      createdAt: new Date(),
      lastActivity: new Date(),
      isActive: true
    };
    setChatSessions([defaultSession]);
  }, []);

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

  // Theme management - initialize from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []); // Only run once on mount

  const applyTheme = React.useCallback((selectedTheme: Theme) => {
    const root = document.documentElement;
    
    if (selectedTheme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', prefersDark);
    } else {
      root.classList.toggle('dark', selectedTheme === 'dark');
    }
    
    // Only update localStorage if it's different to prevent unnecessary updates
    const currentStoredTheme = localStorage.getItem('theme');
    if (currentStoredTheme !== selectedTheme) {
      localStorage.setItem('theme', selectedTheme);
    }
  }, []);

  // Apply theme when theme changes
  useEffect(() => {
    applyTheme(theme);
  }, [theme, applyTheme]);

  const toggleTheme = () => {
    const themes: Theme[] = ['light', 'dark', 'auto'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    setTheme(nextTheme);
  };

  // Agentic AI functions
  const switchAgent = (agentType: string) => {
    setCurrentAgent(agentType);
    setAgentStatus('thinking');
    
    // Add agent switch animation
    setAgentAnimations(prev => ({ ...prev, [agentType]: true }));
    
    setTimeout(() => {
      setAgentStatus('idle');
      setAgentAnimations(prev => ({ ...prev, [agentType]: false }));
    }, 2000);
  };

  const startAgentThinking = (thoughts: string[]) => {
    setAgentThoughts(thoughts);
    setShowThoughtBubble(true);
    setAgentStatus('thinking');
    
    setTimeout(() => {
      setShowThoughtBubble(false);
      setAgentStatus('processing');
    }, 3000);
  };

  const showFloatingAgentAnimation = () => {
    setShowFloatingAgents(true);
    setTimeout(() => setShowFloatingAgents(false), 5000);
  };

  // Get current chat messages
  const currentChat = chatSessions.find(chat => chat.id === currentChatId);
  const messages = currentChat?.messages || [];

  const createNewChat = () => {
    const newChatId = `chat_${Date.now()}`;
    const newSession: ChatSession = {
      id: newChatId,
      title: `Chat ${chatSessions.length + 1}`,
      messages: [],
      createdAt: new Date(),
      lastActivity: new Date(),
      isActive: true
    };
    setChatSessions(prev => [...prev, newSession]);
    setCurrentChatId(newChatId);
  };

  const switchChat = (chatId: string) => {
    setCurrentChatId(chatId);
    setChatSessions(prev => prev.map(chat => ({
      ...chat,
      isActive: chat.id === chatId
    })));
  };

  const handleSendMessage = async (message: string) => {
    // Start agentic thinking animation
    startAgentThinking([
      "Analyzing user request...",
      "Determining best approach...",
      "Preparing response..."
    ]);

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date(),
      chatId: currentChatId
    };

    // Add message to current chat
    setChatSessions(prev => prev.map(chat => 
      chat.id === currentChatId 
        ? {
            ...chat,
            messages: [...chat.messages, userMessage],
            lastActivity: new Date()
          }
        : chat
    ));

    setIsTyping(true);

    try {
      // Analyze message for automation intent
      const isAutomationRequest = message.toLowerCase().includes('automate') || 
                                 message.toLowerCase().includes('book') ||
                                 message.toLowerCase().includes('search') ||
                                 message.toLowerCase().includes('extract') ||
                                 message.toLowerCase().includes('fill') ||
                                 message.toLowerCase().includes('monitor');

      const isSearchRequest = message.toLowerCase().includes('search') ||
                             message.toLowerCase().includes('find') ||
                             message.toLowerCase().includes('look up');

      // Send message to backend
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: currentChatId,
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
          chatId: currentChatId,
          automation: {
            type: isAutomationRequest ? 'workflow_creation' : 'chat_response',
            status: 'running',
            progress: 0
          }
        };

        // Add AI message to current chat
        setChatSessions(prev => prev.map(chat => 
          chat.id === currentChatId 
            ? {
                ...chat,
                messages: [...chat.messages, aiMessage],
                lastActivity: new Date()
              }
            : chat
        ));

        setActiveAutomation(aiMessage.id);

        // Handle different types of requests
        if (isSearchRequest) {
          await executeSearch(message, aiMessage.id);
        } else if (isAutomationRequest) {
          await executeAutomation(message, aiMessage.id);
        } else {
          // Simulate chat response completion
          setTimeout(() => {
            updateMessageInChat(currentChatId, aiMessage.id, {
              automation: { type: 'chat_response', status: 'completed', progress: 100 }
            });
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
        chatId: currentChatId,
        status: 'error'
      };

      setChatSessions(prev => prev.map(chat => 
        chat.id === currentChatId 
          ? {
              ...chat,
              messages: [...chat.messages, errorMessage],
              lastActivity: new Date()
            }
          : chat
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const executeSearch = async (message: string, messageId: string) => {
    try {
      // Start search animation
      startSearchAnimation(messageId);
      
      // Execute enhanced web search
      const searchResponse = await fetch(`${BACKEND_URL}/search/web`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: message, 
          max_results: 15,
          providers: ["google", "bing", "duckduckgo", "github", "stack_overflow", "reddit", "youtube", "news"]
        })
      });

      if (searchResponse.ok) {
        const searchData = await searchResponse.json();
        const results = searchData.results || [];
        
        // Stop search animation
        stopSearchAnimation();
        
        setSearchResults(results);
        setShowSearchResults(true);

        // Update message with enhanced search results
        updateMessageInChat(currentChatId, messageId, {
          automation: { type: 'enhanced_search', status: 'completed', progress: 100 },
          sources: results.map((result: any, index: number) => ({
            title: result.title || `Result ${index + 1}`,
            url: result.url || '#',
            snippet: result.snippet || result.description || '',
            domain: result.domain || 'unknown',
            relevance: result.relevance || 0.8,
            source: result.source || 'web_search',
            confidence: result.confidence || 0.8,
            content_type: result.content_type || 'webpage',
            timestamp: result.timestamp || new Date().toISOString()
          }))
        });

        setActiveAutomation(null);
      } else {
        throw new Error('Enhanced search failed');
      }
    } catch (error) {
      console.error('Enhanced search error:', error);
      stopSearchAnimation();
      updateMessageInChat(currentChatId, messageId, {
        automation: { type: 'enhanced_search', status: 'failed', progress: 0 }
      });
      setActiveAutomation(null);
    }
  };

  const startSearchAnimation = (messageId: string) => {
    let progress = 0;
    const animationInterval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress > 85) progress = 85;
      
      updateMessageInChat(currentChatId, messageId, {
        automation: { 
          type: 'enhanced_search', 
          status: 'running', 
          progress: Math.floor(progress)
        }
      });
    }, 800);
    
    // Store animation interval for cleanup
    if (automationIntervalRef.current) {
      clearInterval(automationIntervalRef.current);
    }
    automationIntervalRef.current = animationInterval;
  };

  const stopSearchAnimation = () => {
    if (automationIntervalRef.current) {
      clearInterval(automationIntervalRef.current);
      automationIntervalRef.current = null;
    }
  };

  const executeAutomation = async (message: string, messageId: string) => {
    try {
      // Start live automation stream
      startLiveAutomationStream(messageId);

      // Determine automation type based on message
      let automationType = 'web_automation';
      let url = '';
      let actions = [];

      if (message.toLowerCase().includes('book') && message.toLowerCase().includes('flight')) {
        // Ticket booking automation
        const bookingResponse = await fetch(`${BACKEND_URL}/automation/ticket-booking`, {
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
          updateMessageInChat(currentChatId, messageId, {
            automation: { 
              type: 'ticket_booking', 
              status: 'completed', 
              progress: 100,
              automationId: bookingData.booking_id,
              screenshots: bookingData.result?.screenshots
            }
          });
          stopLiveAutomationStream();
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

      // Execute intelligent automation
      const automationResponse = await fetch(`${BACKEND_URL}/automation/intelligent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instructions: message,
          url: url || 'https://www.google.com'
        })
      });

      if (automationResponse.ok) {
        const automationData = await automationResponse.json();
        updateMessageInChat(currentChatId, messageId, {
          automation: { 
            type: 'web_automation', 
            status: 'completed', 
            progress: 100,
            automationId: automationData.automation_id,
            screenshots: automationData.result?.screenshots
          }
        });
      } else {
        throw new Error('Automation execution failed');
      }

      stopLiveAutomationStream();
    } catch (error) {
      console.error('Automation error:', error);
      updateMessageInChat(currentChatId, messageId, {
        automation: { type: 'web_automation', status: 'failed', progress: 0 }
      });
      stopLiveAutomationStream();
    }
  };

  const startLiveAutomationStream = (messageId: string) => {
    let progress = 0;
    automationIntervalRef.current = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90;
      
      updateMessageInChat(currentChatId, messageId, {
        automation: { 
          type: 'web_automation', 
          status: 'running', 
          progress: Math.floor(progress)
        }
      });
    }, 1000);
  };

  const stopLiveAutomationStream = () => {
    if (automationIntervalRef.current) {
      clearInterval(automationIntervalRef.current);
      automationIntervalRef.current = null;
    }
  };

  const updateMessageInChat = (chatId: string, messageId: string, updates: Partial<Message>) => {
    setChatSessions(prev => prev.map(chat => 
      chat.id === chatId 
        ? {
            ...chat,
            messages: chat.messages.map(msg => 
              msg.id === messageId 
                ? { ...msg, ...updates }
                : msg
            )
          }
        : chat
    ));
  };

  const handleAutomationControl = (action: string, messageId: string) => {
    console.log(`Automation control: ${action} for message ${messageId}`);
    
    if (action === 'play') {
      setActiveAutomation(messageId);
      startLiveAutomationStream(messageId);
    } else if (action === 'pause') {
      setActiveAutomation(null);
      stopLiveAutomationStream();
      updateMessageInChat(currentChatId, messageId, {
        automation: { type: 'web_automation', status: 'paused', progress: 50 }
      });
    } else if (action === 'handoff') {
      // Request human intervention
      const handoffRequest = {
        id: `handoff_${Date.now()}`,
        messageId,
        reason: 'Requires human input',
        timestamp: new Date(),
        data: { currentStep: 'form_filling', issue: 'captcha_detected' }
      };
      setHandoffRequests(prev => [...prev, handoffRequest]);
      
      updateMessageInChat(currentChatId, messageId, {
        automation: { 
          type: 'web_automation', 
          status: 'handoff_required', 
          progress: 75,
          handoffReason: 'Human intervention required for CAPTCHA'
        }
      });
    }
  };

  const handleTakeover = (handoffId: string, userInput: any) => {
    // Handle human takeover
    console.log('Human takeover:', handoffId, userInput);
    
    // Remove handoff request
    setHandoffRequests(prev => prev.filter(req => req.id !== handoffId));
    
    // Resume automation
    const handoffRequest = handoffRequests.find(req => req.id === handoffId);
    if (handoffRequest) {
      updateMessageInChat(currentChatId, handoffRequest.messageId, {
        automation: { 
          type: 'web_automation', 
          status: 'running', 
          progress: 80,
          takeoverData: userInput
        }
      });
      setActiveAutomation(handoffRequest.messageId);
      startLiveAutomationStream(handoffRequest.messageId);
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
    <div className="h-screen flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
            agentStatus === 'thinking' ? 'animate-agent-thinking bg-gradient-to-r from-blue-400 to-purple-500' : 
            agentStatus === 'processing' ? 'animate-pulse-slow bg-gradient-to-r from-green-400 to-blue-500' :
            'bg-gradient-to-r from-blue-500 to-purple-600'
          }`}>
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white transition-colors">
              Autonomous Automation Platform
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 transition-colors">
              AI-powered workflow automation • {currentAgent.charAt(0).toUpperCase() + currentAgent.slice(1)} Agent Active
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Agent Selector */}
          <div className="relative">
            <button
              onClick={() => setShowAgentSelector(!showAgentSelector)}
              className={`p-2 rounded-lg transition-all duration-300 ${
                showAgentSelector 
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400' 
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
              title="Switch Agent"
            >
              <div className="w-5 h-5 flex items-center justify-center">
                <span className="text-xs font-bold">{currentAgent.charAt(0).toUpperCase()}</span>
              </div>
            </button>
            
            {showAgentSelector && (
              <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50">
                {['planner', 'executor', 'conversational', 'search', 'dom_analysis'].map(agent => (
                  <button
                    key={agent}
                    onClick={() => {
                      switchAgent(agent);
                      setShowAgentSelector(false);
                    }}
                    className={`w-full p-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                      currentAgent === agent ? 'bg-blue-50 dark:bg-blue-900 text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${
                        agent === 'planner' ? 'bg-purple-500' :
                        agent === 'executor' ? 'bg-green-500' :
                        agent === 'conversational' ? 'bg-blue-500' :
                        agent === 'search' ? 'bg-orange-500' :
                        'bg-red-500'
                      }`}></div>
                      <span className="capitalize">{agent} Agent</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={() => setShowSearchResults(!showSearchResults)}
            className={`p-2 rounded-lg transition-all duration-300 ${
              showSearchResults 
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400' 
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title="Toggle Search Results"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
          
          <button
            onClick={() => setShowDashboard(!showDashboard)}
            className={`p-2 rounded-lg transition-all duration-300 ${
              showDashboard 
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400' 
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title="Toggle Dashboard"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
          
          <div className="w-px h-6 bg-gray-300 dark:bg-gray-600"></div>
          
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-300"
            title={`Switch to ${theme === 'light' ? 'dark' : theme === 'dark' ? 'auto' : 'light'} mode`}
          >
            {theme === 'light' ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            ) : theme === 'dark' ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Sessions Sidebar */}
        <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col shadow-sm">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={createNewChat}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-all duration-300 animate-glow"
            >
              + New Chat
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2">
            {chatSessions.map(chat => (
              <div
                key={chat.id}
                onClick={() => switchChat(chat.id)}
                className={`p-3 rounded-lg cursor-pointer transition-all duration-300 ${
                  chat.id === currentChatId 
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100 animate-glow' 
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium truncate">{chat.title}</div>
                <div className={`text-sm truncate ${
                  chat.id === currentChatId ? 'text-blue-700 dark:text-blue-300' : 'text-gray-500 dark:text-gray-400'
                }`}>
                  {chat.messages.length} messages
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Interface */}
        <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900">
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

        {/* Search Results Panel */}
        {showSearchResults && (
          <div className="w-80 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 flex flex-col shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-white">Search Results</h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {searchResults.map((result, index) => (
                <div key={index} className="search-result mb-4 p-3 border border-gray-200 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-all duration-300">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">
                      {result.title}
                    </h4>
                    {result.source && (
                      <span className={`search-result-source source-${result.source}`}>
                        {result.source}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">{result.snippet}</p>
                  <div className="flex items-center justify-between mt-2">
                    <div className="text-xs text-gray-500 dark:text-gray-400">{result.url}</div>
                    {result.confidence && (
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Confidence: {Math.round(result.confidence * 100)}%
                      </div>
                    )}
                  </div>
                  {result.content_type && (
                    <div className="mt-2">
                      <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-full">
                        {result.content_type}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Automation Dashboard */}
        {showDashboard && (
          <div className="w-80 bg-white border-l border-gray-200 shadow-sm">
            <AutomationDashboard
              metrics={automationMetrics}
              agents={agents}
              onAgentControl={handleAgentControl}
              onViewDetails={handleViewDetails}
            />
          </div>
        )}
      </div>

      {/* Handoff Requests */}
      {handoffRequests.length > 0 && (
        <div className="fixed bottom-4 right-4 z-50">
          {handoffRequests.map(request => (
            <div key={request.id} className="bg-yellow-100 border border-yellow-300 rounded-lg p-4 mb-2 max-w-sm shadow-lg">
              <h4 className="font-medium text-yellow-800">Human Intervention Required</h4>
              <p className="text-sm text-yellow-700 mt-1">{request.reason}</p>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => handleTakeover(request.id, { action: 'continue' })}
                  className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 transition-colors"
                >
                  Take Over
                </button>
                <button
                  onClick={() => setHandoffRequests(prev => prev.filter(req => req.id !== request.id))}
                  className="px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-700 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Floating Agents Animation */}
      {showFloatingAgents && (
        <div className="fixed inset-0 pointer-events-none z-40">
          {['planner', 'executor', 'conversational', 'search', 'dom_analysis'].map((agent, index) => (
            <div
              key={agent}
              className={`absolute animate-fade-in-up`}
              style={{
                left: `${20 + (index * 15)}%`,
                top: `${30 + (index * 10)}%`,
                animationDelay: `${index * 0.2}s`
              }}
            >
              <div className={`agent-icon ${
                agent === 'planner' ? 'agent-planner' :
                agent === 'executor' ? 'agent-executor' :
                agent === 'conversational' ? 'agent-conversational' :
                agent === 'search' ? 'agent-search' :
                'agent-dom'
              } animate-agent-thinking`}>
                {agent.charAt(0).toUpperCase()}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Thought Bubble */}
      {showThoughtBubble && (
        <div className="fixed top-20 right-4 z-50 animate-slide-in-right">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-4 max-w-sm">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Thinking...</span>
            </div>
            <div className="space-y-1">
              {agentThoughts.map((thought, index) => (
                <div
                  key={index}
                  className="text-sm text-gray-600 dark:text-gray-400 animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.3}s` }}
                >
                  {thought}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Agent Status Indicator */}
      <div className="fixed bottom-4 left-4 z-50">
        <div className={`px-3 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
          agentStatus === 'thinking' 
            ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 animate-pulse-slow' :
          agentStatus === 'processing'
            ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 animate-pulse-slow' :
            'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
        }`}>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              agentStatus === 'thinking' ? 'bg-blue-500 animate-pulse' :
              agentStatus === 'processing' ? 'bg-green-500 animate-pulse' :
              'bg-gray-500'
            }`}></div>
            <span className="capitalize">{currentAgent} Agent</span>
            <span className="capitalize">• {agentStatus}</span>
          </div>
        </div>
      </div>
    </div>
  );
}