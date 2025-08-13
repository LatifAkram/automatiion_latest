'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { 
  MessageCircle, 
  Plus, 
  X, 
  Send, 
  Mic, 
  Camera, 
  Play, 
  Pause, 
  Download,
  ExternalLink,
  Zap,
  Brain,
  Eye,
  Settings,
  Maximize2,
  Minimize2,
  RefreshCw,
  Search,
  Code,
  Image as ImageIcon,
  Video,
  FileText,
  Globe,
  Sparkles,
  Activity,
  ChevronDown,
  ChevronUp,
  Bot,
  User,
  PlayCircle,
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
  StopCircle
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import { 
  ChatMessage, 
  SearchSource, 
  WorkflowExecution, 
  BrowserSession, 
  AutomationResponse,
  SystemStatus,
  WebSocketMessage 
} from '../app/types';

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  isActive: boolean;
  createdAt: Date;
  lastActivity: Date;
  context?: string;
  type: 'automation' | 'search' | 'hybrid';
}

interface MediaContent {
  type: 'screenshot' | 'video' | 'recording';
  url: string;
  timestamp: Date;
  metadata?: {
    duration?: number;
    size?: { width: number; height: number };
    description?: string;
  };
}

// Utility function to safely convert values to numbers for display
const safeToNumber = (value: any, defaultValue: number = 0): number => {
  if (typeof value === 'number') return isNaN(value) ? defaultValue : value;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? defaultValue : parsed;
  }
  return defaultValue;
};

// Utility function to safely format numbers with toFixed
const safeToFixed = (value: any, decimals: number = 2, defaultValue: string = '0'): string => {
  const num = safeToNumber(value);
  return num.toFixed(decimals);
};

export default function EnhancedChatInterface() {
  // Multi-chat session management
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([
    {
      id: 'default',
      title: 'Agentic AI Assistant',
      messages: [],
      isActive: true,
      createdAt: new Date(),
      lastActivity: new Date(),
      type: 'hybrid'
    }
  ]);
  
  const [activeSessionId, setActiveSessionId] = useState('default');
  const [inputMessage, setInputMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [isTyping, setIsTyping] = useState(false);
  const [typingStatus, setTypingStatus] = useState<Array<{sessionId: string; message: string}>>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [browserSession, setBrowserSession] = useState<BrowserSession | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaContent, setMediaContent] = useState<MediaContent[]>([]);
  const [currentProcessSteps, setCurrentProcessSteps] = useState<Map<string, any[]>>(new Map());
  const [activeProcesses, setActiveProcesses] = useState<Set<string>>(new Set());
  const [processStepsObject, setProcessStepsObject] = useState<{[sessionId: string]: any[]}>({});
  
  // Status popup state
  const [statusPopupVisible, setStatusPopupVisible] = useState(false);
  const [statusPopupMinimized, setStatusPopupMinimized] = useState(false);
  
  // Toast notification state
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastType, setToastType] = useState<'info' | 'warning' | 'error' | 'success'>('info');
  
  // Connection attempt control
  const [shouldAttemptConnection, setShouldAttemptConnection] = useState(true);
  
  // Auto-show status popup when AI processing starts
  useEffect(() => {
    const hasActiveProcessSteps = processStepsObject[activeSessionId] && processStepsObject[activeSessionId].length > 0;
    const hasActiveProcessing = activeProcesses.size > 0 || typingStatus.length > 0;
    
    console.log('🔍 Status popup logic:', {
      activeSessionId,
      hasActiveProcessSteps,
      processStepsCount: processStepsObject[activeSessionId]?.length || 0,
      hasActiveProcessing,
      activeProcessesSize: activeProcesses.size,
      typingStatusLength: typingStatus.length
    });
    
    if (hasActiveProcessSteps || hasActiveProcessing) {
      setStatusPopupVisible(true);
    } else {
      // Auto-hide after processing completes (with delay)
      const timer = setTimeout(() => setStatusPopupVisible(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [processStepsObject, activeSessionId, activeProcesses.size, typingStatus.length]);
  
  // Advanced UI states
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [fullscreenMode, setFullscreenMode] = useState(false);
  const [animationMode, setAnimationMode] = useState<'smooth' | 'fast' | 'minimal'>('smooth');
  const [agenticMode, setAgenticMode] = useState<'autonomous' | 'collaborative' | 'guided'>('autonomous');
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Animation controls
  const chatAnimation = useAnimation();
  const sidebarAnimation = useAnimation();

  const activeSession = chatSessions.find(session => session.id === activeSessionId) || chatSessions[0];

  // WebSocket connection with enhanced features and retry logic
  const connectWebSocket = useCallback(() => {
    // Don't attempt connection if we've determined backend is offline
    if (!shouldAttemptConnection) {
      console.debug('🚫 Skipping WebSocket connection attempt - backend appears to be offline');
      return;
    }

    try {
      setConnectionStatus('connecting');
      const wsUrl = process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:8001';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
        setReconnectAttempts(0);
        setShouldAttemptConnection(true); // Reset connection attempts on successful connection
        console.log('🔌 Enhanced WebSocket connected successfully');
        
        // Send enhanced connection handshake
        ws.send(JSON.stringify({
          type: 'enhanced_connection_with_realtime_agentic_search',
          client_capabilities: {
            multi_chat: true,
            media_support: true,
            real_time_collaboration: true,
            advanced_ui: true,
            version: '2.0'
          },
          session_id: activeSessionId
        }));

        // Load existing sessions from database after connection
        setTimeout(() => loadSessionsFromDatabase(), 1000);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('🔍 Frontend received WebSocket message:', message.type, message);
          
          // FORCE UI UPDATE: Stop thinking indicator when any message received
          if (isTyping) {
            console.log('🔄 Stopping typing indicator due to message received');
            setIsTyping(false);
            setTypingStatus([]);
          }
          
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('WebSocket message parsing error:', error, event.data);
          // Also stop typing indicator on error
          if (isTyping) {
            setIsTyping(false);
            setTypingStatus([]);
          }
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setConnectionStatus('disconnected');
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        
        // Only retry connection if it's not a normal close and we haven't exceeded attempts
        if (event.code !== 1000 && reconnectAttempts < 5 && shouldAttemptConnection) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
          console.log(`🔄 Retrying connection in ${delay/1000}s (attempt ${reconnectAttempts + 1}/5)...`);
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connectWebSocket();
          }, delay);
        } else if (reconnectAttempts >= 5) {
          console.warn('⚠️ Max reconnection attempts reached. Backend may be offline.');
          setConnectionStatus('error');
          setShouldAttemptConnection(false); // Stop future connection attempts
          showToast('Backend server appears to be offline. Please start the server to enable full features.', 'error');
        }
      };

      ws.onerror = (error) => {
        // Only log WebSocket errors for debugging, not user-facing errors
        console.debug('WebSocket connection issue (backend may be offline)');
        setConnectionStatus('error');
        
        // Stop typing indicator on WebSocket error
        if (isTyping) {
          console.log('🔄 Stopping typing indicator due to WebSocket error');
          setIsTyping(false);
          setTypingStatus([]);
        }
        
        // Show user-friendly error message only once per connection attempt
        if (ws.readyState === WebSocket.CONNECTING && reconnectAttempts === 0) {
          console.warn('⚠️ WebSocket connection failed - backend may not be running on ws://localhost:8001');
          console.log('💡 Tip: Start the backend server with: python deepseek-manus-ai-backend.py');
          showToast('Backend server is offline. Some features may be limited.', 'warning');
        } else if (ws.readyState === WebSocket.OPEN) {
          console.warn('⚠️ WebSocket connection interrupted');
          showToast('Connection interrupted. Attempting to reconnect...', 'warning');
        }
      };
      
      wsRef.current = ws;
    } catch (error) {
      console.debug('WebSocket connection attempt failed (backend may be offline)');
      setIsConnected(false);
      setConnectionStatus('error');
      
      // Retry connection after a delay if we haven't exceeded max attempts and should still attempt
      if (reconnectAttempts < 5 && shouldAttemptConnection) {
        const delay = Math.min(2000 * Math.pow(2, reconnectAttempts), 15000);
        console.log(`🔄 Retrying WebSocket connection in ${delay/1000}s...`);
        setTimeout(() => {
          setReconnectAttempts(prev => prev + 1);
          connectWebSocket();
        }, delay);
      } else if (reconnectAttempts >= 5) {
        console.warn('⚠️ Max reconnection attempts reached. Please check if backend is running.');
        setShouldAttemptConnection(false); // Stop future connection attempts
        showToast('Unable to connect to backend server. Please start the server to enable AI features.', 'error');
      }
    }
  }, [activeSessionId, reconnectAttempts, isTyping, shouldAttemptConnection]);

  // Enhanced message handling with multi-session support (DEFENSIVE)
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    // Defensive validation for all incoming WebSocket messages
    if (!message || typeof message !== 'object') {
      console.warn('⚠️ Ignoring malformed WebSocket message:', message);
      return;
    }
    
    if (!message.type || typeof message.type !== 'string') {
      console.warn('⚠️ Ignoring WebSocket message with invalid type:', message);
      return;
    }
    
    const targetSessionId = message.session_id || activeSessionId;
    
    try {
      switch (message.type) {
        case 'enhanced_connection_established':
        case 'enhanced_connection_with_realtime_agentic_search':
          console.log('✅ WebSocket connection established with backend:', message.payload);
          console.log('🔄 Setting connection status to CONNECTED');
          setIsConnected(true);
          setConnectionStatus('connected');
          setReconnectAttempts(0);
          break;
          
        case 'enhanced_automation_result':
        case 'enhanced_automation_with_realtime_search_result':
          handleEnhancedAutomationResult(message.payload, targetSessionId);
          break;
          
        case 'realtime_agentic_search_result':
          handleAgenticSearchResult(message.payload, targetSessionId);
          break;
          
        case 'media_content':
          handleMediaContent(message.payload);
          break;
          
        case 'agentic_user_interaction_needed':
          handleAgenticUserInteraction(message.payload, targetSessionId);
          break;
          
        case 'live_automation_update':
          handleLiveAutomationUpdate(message.payload, targetSessionId);
          break;
          
        case 'output_section_update':
          handleOutputSectionUpdate(message.payload, targetSessionId);
          break;
          
        case 'conversational_commentary':
          handleConversationalCommentary(message.payload, targetSessionId);
          break;
          
        case 'perplexity_display':
          handlePerplexityDisplay(message.payload, targetSessionId);
          break;
          
        case 'browser_session_update':
          setBrowserSession(message.payload);
          break;
          
        case 'system_status':
          setSystemStatus(message.payload);
          break;
          
        case 'typing_indicator':
          setIsTyping(message.payload.isTyping);
          // Update typing status array
          if (message.payload.isTyping) {
            setTypingStatus(prev => {
              const existing = prev.find(status => status.sessionId === targetSessionId);
              if (!existing) {
                return [...prev, { sessionId: targetSessionId, message: 'AI is thinking...' }];
              }
              return prev;
            });
          } else {
            setTypingStatus(prev => prev.filter(status => status.sessionId !== targetSessionId));
          }
          break;
          
        case 'session_created':
          handleNewSession(message.payload);
          break;
          
        case 'process_step_update':
          handleProcessStepUpdate(message.payload, message.session_id);
          break;
          
        case 'chat_sessions_loaded':
          handleSessionsLoaded(message.payload);
          break;
          
        case 'chat_messages_loaded':
          handleMessagesLoaded(message.payload, message.session_id || targetSessionId);
          break;
          
        case 'ai_chat_message':
          console.log('📨 Raw ai_chat_message received:', message);
          console.log('📨 Message session_id:', message.session_id);
          console.log('📨 Target session ID:', targetSessionId);
          console.log('📨 Active session ID:', activeSessionId);
          handleAIChatMessage(message.message, targetSessionId);
          break;
          
        case 'keepalive':
          // Keep-alive message - just log it
          console.log('📡 Keep-alive received:', message.message);
          break;
          
        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('❌ Error handling WebSocket message:', error, { message });
    }
  }, [activeSessionId]);

  // AI Chat Message handler - 🔥 CRITICAL FIX for AI responses not showing
  const handleAIChatMessage = useCallback((aiMessage: any, sessionId: string) => {
    console.log('🤖 AI chat message received:', aiMessage);
    console.log('🔍 Target session ID:', sessionId);
    console.log('🔍 Current chat sessions:', chatSessions);
    
    try {
      // Add the AI message directly to the chat
      setChatSessions(prev => {
        console.log('🔍 Previous sessions:', prev);
        const updated = prev.map(session => {
          if (session.id === sessionId) {
            console.log('✅ Found matching session, adding message:', session.id);
            return { ...session, messages: [...session.messages, aiMessage] };
          }
          return session;
        });
        console.log('🔍 Updated sessions:', updated);
        return updated;
      });
      
      // Stop typing indicator for this session
      setTypingStatus(prev => prev.filter(status => status.sessionId !== sessionId));
      
      console.log('✅ AI message added to chat successfully');
      
    } catch (error) {
      console.error('❌ Error handling AI chat message:', error);
    }
  }, [chatSessions]);

  // Handle loaded sessions from database
  const handleSessionsLoaded = useCallback((sessions: any[]) => {
    console.log('📥 Chat sessions loaded from database:', sessions);
    
    try {
      const loadedSessions: ChatSession[] = sessions.map(session => ({
        id: session.id,
        title: session.title,
        messages: [], // Messages will be loaded separately
        isActive: false,
        createdAt: new Date(session.created_at),
        lastActivity: new Date(session.last_activity),
        type: session.type || 'hybrid'
      }));
      
      if (loadedSessions.length > 0) {
        setChatSessions(loadedSessions);
        // Set the most recent session as active
        const mostRecentSession = loadedSessions.sort((a, b) => 
          b.lastActivity.getTime() - a.lastActivity.getTime()
        )[0];
        setActiveSessionId(mostRecentSession.id);
        
        // Load messages for the active session
        setTimeout(() => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'load_chat_messages',
              session_id: mostRecentSession.id
            }));
          }
        }, 100);
      }
    } catch (error) {
      console.error('❌ Error handling loaded sessions:', error);
    }
  }, []);

  // Agentic user interaction handler
  const handleAgenticUserInteraction = useCallback((payload: any, sessionId: string) => {
    console.log('🤝 Agentic AI requesting user interaction:', payload);
    
    try {
      const interactionMessage: ChatMessage = {
        id: `agentic-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'ai_response',
        content: `${payload.message}\n\n**I need the following information:**\n${payload.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}\n\n*Please provide your answers so I can help you better with this ${payload.automation_type} task.*`,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        metadata: {
          interaction_needed: true,
          automation_type: payload.automation_type,
          questions: payload.questions,
          ai_provider: 'agentic_interaction',
          requires_user_input: true
        }
      };
      
      // Add the interaction message to chat
      setChatSessions(prev => prev.map(session => 
        session.id === sessionId 
          ? { ...session, messages: [...session.messages, interactionMessage] }
          : session
      ));
      
      // Stop typing indicator since we need user input
      setIsTyping(false);
      
      console.log('✅ Agentic interaction message added to chat');
      
    } catch (error) {
      console.error('❌ Error handling agentic user interaction:', error);
    }
  }, []);

  const handleLiveAutomationUpdate = useCallback((payload: any, sessionId: string) => {
    console.log('🎬 Live automation update:', payload);
    
    try {
      const updateMessage: ChatMessage = {
        id: `live-update-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'ai_response',
        content: `**🎬 ${payload.step_name || 'Live Automation Update'}**

${payload.message}

${payload.details ? `*${payload.details}*` : ''}

${payload.media_captured ? '📸 *Screenshot captured*' : ''}
${payload.video_recording ? '🎥 *Video recording in progress*' : ''}`,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        metadata: {
          update_type: 'live_automation',
          step_number: payload.step_number,
          total_steps: payload.total_steps,
          status: payload.status,
          media_captured: payload.media_captured || false
        }
      };
      
      setChatSessions(prev => prev.map(session => 
        session.id === sessionId 
          ? { ...session, messages: [...session.messages, updateMessage] }
          : session
      ));
      
    } catch (error) {
      console.error('❌ Error handling live automation update:', error);
    }
  }, []);

  const handleOutputSectionUpdate = useCallback((payload: any, sessionId: string) => {
    console.log('📊 Output section update:', payload);
    
    try {
      const outputMessage: ChatMessage = {
        id: `output-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'ai_response',
        content: `## 📊 **Automation Complete - Comprehensive Report**

### 🎬 **Live Session Summary**
- **Session ID:** \`${payload.session_summary?.session_id || 'N/A'}\`
- **Duration:** ${payload.session_summary?.total_duration ? `${Math.round(payload.session_summary.total_duration)}s` : 'N/A'}
- **Steps Executed:** ${payload.session_summary?.steps_executed || 0}
- **Success Rate:** ${payload.session_summary?.success_rate ? `${Math.round(payload.session_summary.success_rate)}%` : 'N/A'}

### 📸 **Media Gallery**
- **Screenshots:** ${payload.media_gallery?.screenshots?.length || 0} captured
- **Videos:** ${payload.media_gallery?.videos?.length || 0} recorded
- **Total Media Files:** ${payload.media_gallery?.total_media_files || 0}

### 💻 **Generated Code Available**
- **Playwright:** ${payload.generated_code?.playwright?.length || 0} functions
- **Selenium:** ${payload.generated_code?.selenium?.length || 0} functions  
- **Cypress:** ${payload.generated_code?.cypress?.length || 0} tests

### 🌐 **Web Research Summary**
- **Sources Found:** ${payload.execution_report?.web_research_summary?.sources_found || 0}
- **Live Data Points:** ${payload.execution_report?.web_research_summary?.live_data_points || 0}
- **Confidence Score:** ${payload.execution_report?.web_research_summary?.confidence_score ? `${Math.round(payload.execution_report.web_research_summary.confidence_score * 100)}%` : 'N/A'}

### 🤝 **Human Interactions**
- **Takeovers:** ${payload.human_interactions?.length || 0}
- **Learning Data:** ${payload.execution_report?.learning_insights?.length || 0} points

*All media files, generated code, and detailed reports are available for download.*

**🏆 System Performance:** Superior to Manus AI and enterprise RPA tools!`,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        metadata: {
          output_section: payload,
          has_media: (payload.media_gallery?.total_media_files || 0) > 0,
          has_code: Object.values(payload.generated_code || {}).some((arr: any) => Array.isArray(arr) && arr.length > 0),
          automation_type: 'comprehensive_report'
        }
      };
      
      setChatSessions(prev => prev.map(session => 
        session.id === sessionId 
          ? { ...session, messages: [...session.messages, outputMessage] }
          : session
      ));
      
    } catch (error) {
      console.error('❌ Error handling output section update:', error);
    }
  }, []);

  const handleConversationalCommentary = useCallback((payload: any, sessionId: string) => {
    console.log('🤖 Conversational commentary:', payload);
    
    try {
      const commentaryMessage: ChatMessage = {
        id: `commentary-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'ai_response',
        content: `🤖 **${payload.commentary || payload.message}**

${payload.reasoning ? `*Reasoning: ${payload.reasoning}*` : ''}`,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        metadata: {
          commentary_type: 'cursor_ai_style',
          step_number: payload.step_number,
          commentary_category: payload.type || 'general'
        }
      };
      
      setChatSessions(prev => prev.map(session => 
        session.id === sessionId 
          ? { ...session, messages: [...session.messages, commentaryMessage] }
          : session
      ));
      
    } catch (error) {
      console.error('❌ Error handling conversational commentary:', error);
    }
  }, []);

  const handlePerplexityDisplay = useCallback((payload: any, sessionId: string) => {
    console.log('🌐 Perplexity-style web research display:', payload);
    
    try {
      // Create Perplexity-style web research display
      let researchContent = `## 🌐 **Web Research Results** (Perplexity AI-style)\n\n`;
      
      // Display sources
      if (payload.sources && payload.sources.length > 0) {
        researchContent += `### 📚 **Sources Found:** ${payload.sources.length}\n\n`;
        payload.sources.forEach((source: any, index: number) => {
          researchContent += `**${index + 1}.** [${source.title || 'Source'}](${source.url})\n`;
          if (source.snippet) {
            researchContent += `   *${source.snippet}*\n`;
          }
          researchContent += `   **Confidence:** ${Math.round((source.confidence || 0.5) * 100)}% | **Provider:** ${source.source || 'Unknown'}\n\n`;
        });
      }
      
      // Display live data
      if (payload.live_data && Object.keys(payload.live_data).length > 0) {
        researchContent += `### 📊 **Live Data Extracted:**\n\n`;
        Object.entries(payload.live_data).forEach(([key, value]) => {
          researchContent += `- **${key.replace('_', ' ').toUpperCase()}:** ${value}\n`;
        });
        researchContent += `\n`;
      }
      
      // Display AI insights
      if (payload.ai_insights && payload.ai_insights.recommendations) {
        researchContent += `### 🤖 **AI Insights:**\n\n`;
        payload.ai_insights.recommendations.forEach((rec: string, index: number) => {
          researchContent += `${index + 1}. ${rec}\n`;
        });
        researchContent += `\n**Data Quality:** ${payload.ai_insights.data_quality || 'Unknown'}\n`;
        researchContent += `**Confidence Level:** ${Math.round((payload.ai_insights.confidence_level || 0) * 100)}%\n\n`;
      }
      
      researchContent += `*Research completed using multiple providers for comprehensive coverage*`;
      
      const researchMessage: ChatMessage = {
        id: `research-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'ai_response',
        content: researchContent,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        metadata: {
          display_type: 'perplexity_style',
          sources_count: payload.sources?.length || 0,
          live_data_points: Object.keys(payload.live_data || {}).length,
          research_quality: payload.ai_insights?.data_quality || 'unknown'
        }
      };
      
      setChatSessions(prev => prev.map(session => 
        session.id === sessionId 
          ? { ...session, messages: [...session.messages, researchMessage] }
          : session
      ));
      
      console.log('✅ Perplexity-style research results displayed');
      
    } catch (error) {
      console.error('❌ Error handling Perplexity display:', error);
    }
  }, []);

  // Enhanced automation result handler
  const handleEnhancedAutomationResult = useCallback((payload: any, sessionId: string) => {
    console.log('🤖 Enhanced automation result received:', payload);
    
    // Create AI response message with proper structure for sources display
    const aiMessage: ChatMessage = {
      id: `ai-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'ai_response',
      content: payload.success 
        ? `✅ Automation completed successfully!\n\n${payload.competitive_advantages?.join('\n• ') || 'Task completed with superior performance.'}` 
        : `❌ Automation failed: ${payload.error || 'Unknown error'}`,
      timestamp: new Date(),
      ai_response: {
        main_answer: payload.success ? 'Automation completed successfully' : 'Automation failed',
        confidence: payload.performance_score ? payload.performance_score / 100 : 0.8,
        reasoning: payload.competitive_advantages?.join(', ') || 'Advanced automation execution',
        // Extract AI analysis from payload
        ai_analysis: payload.ai_analysis ? {
          analysis: payload.ai_analysis.analysis || 'AI analysis completed',
          plan: payload.ai_analysis.plan || [],
          complexity: payload.ai_analysis.complexity || 'moderate',
          requires_search: payload.ai_analysis.requires_search || false,
          requires_human: payload.ai_analysis.requires_human || false,
          confidence: payload.ai_analysis.confidence || 0.7,
          ai_provider: payload.ai_analysis.ai_provider || 'unknown'
        } : undefined,
        // Extract sources from agentic search results
        sources: payload.real_time_agentic_search_details?.sources || []
      },
      execution: {
        id: `exec-${Date.now()}`,
        workflow_id: 'automation',
        status: payload.success ? 'completed' : 'failed',
        progress: 100,
        steps: [],
        start_time: new Date().toISOString(),
        end_time: new Date().toISOString(),
        duration: payload.execution_time || 0,
        success_rate: payload.success_rate || 0,
        actions_executed: payload.actions_executed || 0
      }
    };
    
    updateSessionMessages(sessionId, aiMessage);
    setIsTyping(false);
  }, []);

  // Agentic search result handler  
  const handleAgenticSearchResult = useCallback((payload: any, sessionId: string) => {
    console.log('🔍 Agentic search result received:', payload);
    
    const aiMessage: ChatMessage = {
      id: `search-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'ai_response', 
      content: payload.conversational_response?.summary || 'Search completed successfully',
      timestamp: new Date(),
      ai_response: {
        main_answer: payload.conversational_response?.summary || 'Search results found',
        confidence: payload.conversational_response?.confidence || 0.8,
        reasoning: 'Agentic search with multiple providers',
        sources: payload.sources || []
      }
    };
    
    updateSessionMessages(sessionId, aiMessage);
  }, []);

  // Process step updates handler (DEFENSIVE PROGRAMMING)
  const handleProcessStepUpdate = useCallback((payload: any, targetSessionId?: string) => {
    // Add debugging to see what data we're receiving
    console.log('🔍 Process step update received:', { payload, targetSessionId });
    
    // Comprehensive defensive validation
    if (!payload || typeof payload !== 'object') {
      console.warn('⚠️ Ignoring process step update - invalid payload:', payload);
      return;
    }
    
    const sessionId = targetSessionId || activeSessionId;
    const step = payload.step || payload;
    
    if (!step || typeof step !== 'object') {
      console.warn('⚠️ Ignoring process step update - invalid step:', { payload, step });
      return;
    }
    
    const stepId = step.id;
    const stepStatus = step.status;
    
    if (!stepId || typeof stepId !== 'string') {
      console.warn('⚠️ Ignoring process step - missing or invalid step.id:', { step, stepId });
      return;
    }
    
    if (!stepStatus || typeof stepStatus !== 'string') {
      console.warn('⚠️ Ignoring process step - missing or invalid step.status:', { step, stepStatus });
      return;
    }
    
    try {
      // Update process steps for the session (defensive array handling)
      // Use both Map and Object approaches for maximum compatibility
      setCurrentProcessSteps(prev => {
        const newSteps = new Map(prev);
        const sessionSteps = newSteps.get(sessionId) || [];
        
        // Safely find existing step index (handle undefined elements)
        const existingIndex = sessionSteps.findIndex(s => s?.id === stepId);
        
        if (existingIndex >= 0) {
          // Update existing step (defensive copy)
          const updatedSteps = [...sessionSteps];
          updatedSteps[existingIndex] = { ...step };
          newSteps.set(sessionId, updatedSteps);
          console.log(`🔄 Updated existing step ${existingIndex} for session ${sessionId}:`, updatedSteps);
        } else {
          // Add new step (defensive append)
          const newSteps_array = [...sessionSteps, { ...step }];
          newSteps.set(sessionId, newSteps_array);
          console.log(`➕ Added new step to session ${sessionId}:`, newSteps_array);
        }
        
        console.log(`📊 Total steps for session ${sessionId}:`, newSteps.get(sessionId)?.length || 0);
        return newSteps;
      });
      
      // ALSO update object version for React-friendly rendering
      setProcessStepsObject(prev => {
        const sessionSteps = prev[sessionId] || [];
        const existingIndex = sessionSteps.findIndex(s => s?.id === stepId);
        
        if (existingIndex >= 0) {
          const updatedSteps = [...sessionSteps];
          updatedSteps[existingIndex] = { ...step };
          return { ...prev, [sessionId]: updatedSteps };
        } else {
          return { ...prev, [sessionId]: [...sessionSteps, { ...step }] };
        }
      });
      
      // Track active processes (defensive Set operations)
      if (stepStatus === 'in_progress') {
        setActiveProcesses(prev => new Set(Array.from(prev).concat(sessionId)));
      } else if (stepStatus === 'completed' || stepStatus === 'failed') {
        // Check if all steps for this session are done
        const allSteps = currentProcessSteps.get(sessionId) || [];
        const activeSteps = allSteps.filter(s => s?.status === 'in_progress');
        if (activeSteps.length <= 1) { // This step + possibly others completing
          setActiveProcesses(prev => {
            const newSet = new Set(Array.from(prev));
            newSet.delete(sessionId);
            return newSet;
          });
        }
      }
    } catch (error) {
      console.error('❌ Error in handleProcessStepUpdate:', error, { payload, step, sessionId });
    }
  }, [activeSessionId, currentProcessSteps]);

  // Session management handlers
  const handleMessagesLoaded = useCallback((messages: any[], sessionId: string) => {
    if (Array.isArray(messages)) {
      const loadedMessages: ChatMessage[] = messages.map(m => ({
        id: m.id || Date.now().toString(),
        type: m.type || 'ai_response',
        content: m.content || '',
        timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
        ai_response: m.ai_response_data ? JSON.parse(m.ai_response_data) : undefined,
        execution: m.execution_data ? JSON.parse(m.execution_data) : undefined,
        media: m.media_data ? JSON.parse(m.media_data) : undefined
      }));
      
      setChatSessions(prev =>
        prev.map(session =>
          session.id === sessionId
            ? { ...session, messages: loadedMessages }
            : session
        )
      );
    }
  }, []);

  // Media content handler (DEFENSIVE)
  const handleMediaContent = useCallback((mediaData: any) => {
    try {
      if (!mediaData || typeof mediaData !== 'object') {
        console.warn('⚠️ Ignoring invalid media content:', mediaData);
        return;
      }
      
      const newMedia: MediaContent = {
        type: mediaData.type || 'screenshot',
        url: mediaData.url || '',
        timestamp: new Date(),
        metadata: mediaData.metadata || {}
      };
      
      setMediaContent(prev => [...prev, newMedia]);
    } catch (error) {
      console.error('❌ Error handling media content:', error, { mediaData });
    }
  }, []);

  // New session handler (DEFENSIVE)
  const handleNewSession = useCallback((sessionData: any) => {
    try {
      if (!sessionData || typeof sessionData !== 'object') {
        console.warn('⚠️ Ignoring invalid session data:', sessionData);
        return;
      }
      
      const newSession: ChatSession = {
        id: sessionData.id || Date.now().toString(),
        title: sessionData.title || 'New Session',
        messages: [],
        isActive: false,
        createdAt: new Date(),
        lastActivity: new Date(),
        type: sessionData.type || 'hybrid'
      };
      
      setChatSessions(prev => [...prev, newSession]);
      setActiveSessionId(newSession.id);
      
      // 🔥 IMPROVED: Only save to database if WebSocket is properly connected
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && isConnected) {
        try {
          wsRef.current.send(JSON.stringify({
            type: 'save_chat_session',
            session_data: {
              id: newSession.id,
              title: newSession.title,
              type: newSession.type,
              created_at: newSession.createdAt.toISOString(),
              last_activity: newSession.lastActivity.toISOString(),
              message_count: 0,
              user_id: 'default_user',
              agentic_mode: agenticMode,
              metadata: JSON.stringify({ ui_version: '2.0', created_via: 'new_chat_button' })
            }
          }));
          console.log('💾 New session saved to database:', newSession.id);
        } catch (error) {
          console.error('❌ Failed to save new session:', error);
          // Don't retry connection here - let the main reconnection logic handle it
        }
      } else {
        console.warn('⚠️ WebSocket not connected - new session created locally only');
        // Show user-friendly message that the session will be saved when connection is restored
        console.log('📝 Session will be saved to database when connection is restored');
      }
    } catch (error) {
      console.error('❌ Error handling new session:', error, { sessionData });
    }
  }, [agenticMode, isConnected]);

  // Session management functions
  const truncateCurrentSession = useCallback(() => {
    console.log(`✂️ Current session ${activeSessionId} processing truncated`);
    setIsTyping(false);
    
    // Clear active processes
    setActiveProcesses(new Set());
    
    // Clear process steps for current session
    setCurrentProcessSteps(prev => {
      const newSteps = new Map(prev);
      newSteps.delete(activeSessionId);
      return newSteps;
    });
    
    setProcessStepsObject(prev => {
      const updated = { ...prev };
      delete updated[activeSessionId];
      return updated;
    });
    
    // Add a truncation message
    const truncationMessage: ChatMessage = {
      id: `truncate-${Date.now()}`,
      type: 'ai_response',
      content: '⚠️ Processing was stopped by user. You can ask me anything else!',
      timestamp: new Date(),
      ai_response: {
        main_answer: 'Processing truncated',
        confidence: 1.0,
        reasoning: 'User requested to stop processing'
      }
    };
    
    updateSessionMessages(activeSessionId, truncationMessage);
  }, [activeSessionId]);

  const deleteSession = useCallback((sessionId: string) => {
    if (chatSessions.length <= 1) {
      console.warn('⚠️ Cannot delete the last session');
      return;
    }
    
    setChatSessions(prev => {
      const updatedSessions = prev.filter(session => session.id !== sessionId);
      
      // If we deleted the active session, switch to another one
      if (sessionId === activeSessionId) {
        const newActiveSession = updatedSessions[0];
        setActiveSessionId(newActiveSession.id);
      }
      
      return updatedSessions;
    });
    
    console.log(`🗑️ Deleted session: ${sessionId}`);
  }, [chatSessions.length, activeSessionId]);

  // Toast notification function
  const showToast = useCallback((message: string, type: 'info' | 'warning' | 'error' | 'success' = 'info') => {
    setToastMessage(message);
    setToastType(type);
    
    // Auto-hide toast after 5 seconds
    setTimeout(() => {
      setToastMessage(null);
    }, 5000);
  }, []);

  const createNewSession = useCallback((type: 'automation' | 'search' | 'hybrid' = 'hybrid') => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: `${type === 'automation' ? '🤖' : type === 'search' ? '🔍' : '🧠'} New Session`,
      messages: [],
      isActive: false,
      createdAt: new Date(),
      lastActivity: new Date(),
      type
    };

    setChatSessions(prev => [...prev, newSession]);
    setActiveSessionId(newSession.id);
    
    // 🔥 IMPROVED: Only save to database if WebSocket is properly connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && isConnected) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'save_chat_session',
          session_data: {
            id: newSession.id,
            title: newSession.title,
            type: newSession.type,
            created_at: newSession.createdAt.toISOString(),
            last_activity: newSession.lastActivity.toISOString(),
            message_count: 0,
            user_id: 'default_user',
            agentic_mode: agenticMode,
            metadata: JSON.stringify({ ui_version: '2.0', created_via: 'new_chat_button' })
          }
        }));
        console.log('💾 New session saved to database:', newSession.id);
        showToast('New chat session created successfully!', 'success');
      } catch (error) {
        console.error('❌ Failed to save new session:', error);
        showToast('Session created locally. Will sync when connection is restored.', 'warning');
      }
    } else {
      console.warn('⚠️ WebSocket not connected - new session created locally only');
      showToast('Session created locally. Backend connection required for full features.', 'warning');
      console.log('📝 Session will be saved to database when connection is restored');
    }
  }, [agenticMode, isConnected, showToast]);

  // Update session messages
  const updateSessionMessages = useCallback((sessionId: string, message: ChatMessage) => {
    setChatSessions(prev => 
      prev.map(session => 
        session.id === sessionId
          ? { ...session, messages: [...session.messages, message], lastActivity: new Date() }
          : session
      )
    );
  }, []);

  const saveSessionToDatabase = useCallback((session: ChatSession) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'save_chat_session',
        session: {
          id: session.id,
          title: session.title,
          type: session.type,
          created_at: session.createdAt.toISOString(),
          last_activity: session.lastActivity.toISOString(),
          message_count: session.messages.length,
          user_id: 'default_user',
          agentic_mode: agenticMode,
          metadata: JSON.stringify({ ui_version: '2.0' })
        }
      }));
    }
  }, [agenticMode]);

  const loadSessionsFromDatabase = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'load_chat_sessions',
        user_id: 'default_user'
      }));
    }
  }, []);

  // Enhanced message sending
  const sendEnhancedMessage = useCallback(async () => {
    if (!inputMessage.trim()) {
      return;
    }

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      showToast('Backend connection required. Please wait for connection or start the backend server.', 'warning');
      return;
    }

    // Add user message to current session
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    updateSessionMessages(activeSessionId, userMessage);

    // Determine message type based on content and agentic mode
    const isSearchRequest = inputMessage.toLowerCase().includes('search') || 
                           inputMessage.toLowerCase().includes('find') ||
                           inputMessage.toLowerCase().includes('look up');

    const enhancedRequest = {
      type: isSearchRequest ? 'direct_realtime_agentic_search' : 'enhanced_automation_with_realtime_agentic_fallback',
      payload: !isSearchRequest ? {
        instruction: inputMessage,
        session_id: activeSessionId,
        conversation_id: activeSessionId,
        context: {
          session_type: activeSession.type,
          previous_messages: activeSession.messages.slice(-5),
          agentic_mode: agenticMode,
          user_preferences: {
            automation_level: agenticMode,
            ui_updates: true,
            real_time_feedback: true
          }
        },
        capabilities_requested: {
          real_time_feedback: true,
          media_capture: true,
          advanced_reasoning: true,
          multi_step_planning: true
        }
      } : undefined,
      query: isSearchRequest ? inputMessage : undefined,
      instruction: !isSearchRequest ? inputMessage : undefined,
      session_id: activeSessionId,
      conversation_id: activeSessionId,
      context: {
        previous_messages: activeSession.messages.slice(-5),
        session_type: activeSession.type,
        agentic_mode: agenticMode
      }
    };

    console.log('🚀 Sending message to backend:', enhancedRequest);
    wsRef.current.send(JSON.stringify(enhancedRequest));
    
    // 🔥 CRITICAL FIX: Save user message to database
    const dbUserMessage = {
      id: `user-${Date.now()}`,
      type: 'user_message',
      content: inputMessage,
      timestamp: new Date().toISOString(),
      session_id: activeSessionId
    };
    
    try {
      wsRef.current.send(JSON.stringify({
        type: 'save_chat_message',
        message_data: dbUserMessage
      }));
      console.log('💾 User message saved to database');
    } catch (error) {
      console.error('❌ Failed to save user message:', error);
    }
    
    setInputMessage('');
    setIsTyping(true);
  }, [inputMessage, activeSessionId, activeSession, agenticMode, showToast, updateSessionMessages]);

  // Load chat sessions from database
  const loadChatSessions = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'load_chat_sessions',
          user_id: 'default_user'
        }));
        console.log('📥 Loading chat sessions from database...');
      } catch (error) {
        console.error('❌ Failed to load chat sessions:', error);
      }
    }
  }, []);

  // Initialize connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);
  
  // Load sessions when connected
  useEffect(() => {
    if (isConnected) {
      // Small delay to ensure connection is fully established
      setTimeout(() => {
        loadChatSessions();
      }, 500);
    }
  }, [isConnected, loadChatSessions]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession.messages]);

  // Enhanced animations
  const messageVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: { 
        type: "spring", 
        stiffness: 500, 
        damping: 30 
      }
    },
    exit: { 
      opacity: 0, 
      y: -20, 
      scale: 0.95,
      transition: { duration: 0.2 }
    }
  };

  const sidebarVariants = {
    expanded: { width: '320px', opacity: 1 },
    collapsed: { width: '60px', opacity: 0.8 }
  };

  // Render enhanced message with media support
  const renderEnhancedMessage = (message: ChatMessage) => (
    <motion.div
      key={message.id}
      variants={messageVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} mb-6`}
    >
      <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
        {/* Avatar */}
        <div className={`flex items-start gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
          <motion.div
            className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold ${
              message.type === 'user' ? 'bg-gradient-to-r from-blue-600 to-purple-600' : 'bg-gradient-to-r from-emerald-500 to-teal-600'
            }`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {message.type === 'user' ? <User size={20} /> : <Bot size={20} />}
          </motion.div>
          
          <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                {message.type === 'user' ? 'You' : 'Agentic AI'}
              </span>
              <span className="text-xs text-gray-500">
                {typeof message.timestamp === 'string' ? new Date(message.timestamp).toLocaleTimeString() : message.timestamp.toLocaleTimeString()}
              </span>
            </div>
            
            {/* Message Content */}
            <motion.div
              className={`rounded-2xl p-4 ${
                message.type === 'user' 
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white' 
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
              }`}
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <div className="prose dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code: ({ className, children, ...props }: any) => {
                      const isInline = !className;
                      return isInline ? (
                        <code className="bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded text-sm" {...props}>
                          {children}
                        </code>
                      ) : (
                        <SyntaxHighlighter
                          style={vscDarkPlus as any}
                          language={className?.replace('language-', '') || 'text'}
                          PreTag="div"
                          className="rounded-lg"
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      );
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
              
              {/* AI Analysis Display (NEW) */}
              {message.type !== 'user' && message.ai_response?.ai_analysis && (
                <motion.div 
                  className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg border-l-4 border-blue-500"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="text-blue-600" size={16} />
                    <span className="font-semibold text-blue-800 dark:text-blue-300 text-sm">
                      AI Analysis ({message.ai_response.ai_analysis.ai_provider || 'rule_based'})
                    </span>
                    <div className="ml-auto text-xs text-blue-600 dark:text-blue-400">
                      {safeToFixed(safeToNumber(message.ai_response.ai_analysis.confidence) * 100, 0)}% confidence
                    </div>
                  </div>
                  <p className="text-sm text-blue-700 dark:text-blue-200 mb-2">
                    {message.ai_response.ai_analysis.analysis}
                  </p>
                  {message.ai_response.ai_analysis.plan && (
                    <div className="text-xs text-blue-600 dark:text-blue-300">
                      <strong>Plan:</strong> {message.ai_response.ai_analysis.plan.length} steps • 
                      Complexity: {message.ai_response.ai_analysis.complexity} • 
                      {message.ai_response.ai_analysis.requires_search ? 'Search enhanced' : 'Direct execution'}
                    </div>
                  )}
                </motion.div>
              )}
              
              {/* Agentic Search Sources Display (NEW - Like Perplexity AI) */}
              {message.type !== 'user' && message.ai_response?.sources && message.ai_response.sources.length > 0 && (
                <motion.div 
                  className="mt-4"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ delay: 0.5 }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <ExternalLink className="text-emerald-600" size={16} />
                    <span className="font-semibold text-emerald-800 dark:text-emerald-300 text-sm">
                      Web Resources ({message.ai_response.sources.length})
                    </span>
                  </div>
                  <div className="grid gap-2">
                    {message.ai_response.sources.slice(0, 5).map((source: any, index: number) => (
                      <motion.div
                        key={index}
                        className="p-3 bg-emerald-50 dark:bg-emerald-900/30 rounded-lg border border-emerald-200 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors cursor-pointer"
                        whileHover={{ scale: 1.02 }}
                        onClick={() => window.open(source.url, '_blank')}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <h4 className="font-medium text-sm text-emerald-800 dark:text-emerald-200 truncate">
                              {source.title}
                            </h4>
                            <p className="text-xs text-emerald-600 dark:text-emerald-300 mt-1 line-clamp-2">
                              {source.content}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <span className="text-xs text-emerald-500 dark:text-emerald-400">
                                {source.source}
                              </span>
                              {source.relevance_score && (
                                <div className="flex items-center gap-1">
                                  <div className="w-12 h-1 bg-emerald-200 dark:bg-emerald-700 rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-emerald-500 transition-all duration-300"
                                      style={{ width: `${safeToNumber(source.relevance_score) * 100}%` }}
                                    />
                                  </div>
                                                                      <span className="text-xs text-emerald-500">
                                      {safeToFixed(safeToNumber(source.relevance_score) * 100, 0)}%
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                          <ExternalLink className="text-emerald-500 flex-shrink-0" size={14} />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
              
                             {/* Media Content Display */}
               {message.type !== 'user' && message.media && (
                 <motion.div 
                   className="mt-4"
                   initial={{ opacity: 0, scale: 0.9 }}
                   animate={{ opacity: 1, scale: 1 }}
                   transition={{ delay: 0.4 }}
                 >
                   {message.media.type === 'screenshot' && (
                     <div className="space-y-2">
                       <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Screenshot:</span>
                                               <motion.img
                          src={message.media.url}
                          alt="Screenshot"
                          className="rounded-lg border border-gray-200 dark:border-gray-600 cursor-pointer hover:scale-105 transition-transform max-w-full"
                          whileHover={{ scale: 1.05 }}
                          onClick={() => message.media && window.open(message.media.url, '_blank')}
                        />
                     </div>
                   )}
                   
                   {message.media.type === 'video' && (
                     <div className="space-y-2">
                       <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Video:</span>
                       <video
                         src={message.media.url}
                         controls
                         className="rounded-lg border border-gray-200 dark:border-gray-600 w-full"
                       />
                     </div>
                   )}
                   
                   {message.media.type === 'recording' && (
                     <div className="space-y-2">
                       <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Recording:</span>
                       <video
                         src={message.media.url}
                         controls
                         className="rounded-lg border border-gray-200 dark:border-gray-600 w-full"
                       />
                     </div>
                   )}
                 </motion.div>
               )}
              
              {/* Execution Details */}
              {message.type !== 'user' && message.execution && (
                <motion.div 
                  className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 }}
                >
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Activity size={16} />
                    Execution Analytics
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-300">Status:</span>
                      <div className={`font-medium ${
                        message.execution.status === 'completed' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {message.execution.status}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-300">Duration:</span>
                      <div className="font-medium">{safeToFixed(message.execution.duration, 2)}s</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-300">Success Rate:</span>
                      <div className="font-medium">{safeToFixed(safeToNumber(message.execution.success_rate) * 100, 0)}%</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-300">Actions:</span>
                      <div className="font-medium">{message.execution.actions_executed || 0}</div>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className={`flex h-screen bg-gray-50 dark:bg-gray-900 ${fullscreenMode ? 'fixed inset-0 z-50' : ''}`}>
      {/* Enhanced Sidebar */}
      <motion.div
        className="bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col"
        variants={sidebarVariants}
        animate={sidebarCollapsed ? 'collapsed' : 'expanded'}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            {!sidebarCollapsed && (
              <motion.h2 
                className="text-lg font-bold text-gray-800 dark:text-gray-200"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                Agentic AI
              </motion.h2>
            )}
            <motion.button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              {sidebarCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
            </motion.button>
          </div>


        </div>

        {/* New Chat Button */}
        {!sidebarCollapsed && (
          <motion.div 
            className="p-4"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <motion.button
              onClick={() => createNewSession('hybrid')}
              className="w-full flex items-center gap-3 p-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Plus size={20} />
              New Agentic Session
            </motion.button>
          </motion.div>
        )}

        {/* Chat Sessions */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence>
            {chatSessions.map((session) => (
              <motion.div
                key={session.id}
                className={`mx-2 mb-2 rounded-lg transition-all group ${
                  session.id === activeSessionId
                    ? 'bg-blue-50 dark:bg-blue-900 border-l-4 border-blue-600'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <div className="flex items-center">
                  <div 
                    className="flex-1 p-3 cursor-pointer"
                    onClick={() => setActiveSessionId(session.id)}
                  >
                    {!sidebarCollapsed && (
                      <div>
                        <div className="font-medium text-sm truncate">{session.title}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {session.messages.length} messages • {session.type}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Delete button */}
                  {!sidebarCollapsed && chatSessions.length > 1 && (
                    <motion.button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession(session.id);
                      }}
                      className="p-2 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900 opacity-0 group-hover:opacity-100 transition-all mr-2"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      title="Delete session"
                    >
                      <X size={16} />
                    </motion.button>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Enhanced Header */}
        <motion.div 
          className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.div 
                className="flex items-center gap-2"
                whileHover={{ scale: 1.05 }}
              >
                <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-full flex items-center justify-center">
                  <Brain size={16} color="white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold">{activeSession.title}</h1>
                  <p className="text-sm text-gray-500">
                    {activeSession.type} • {activeSession.messages.length} messages
                  </p>
                </div>
              </motion.div>

              {/* Real-time process steps display */}
              {(currentProcessSteps.get(activeSessionId)?.length || 0) > 0 && (
                <motion.div 
                  className="flex items-center gap-3 text-sm"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <div className="flex items-center gap-2">
                    <motion.div 
                      className="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    />
                    <motion.div 
                      className="flex flex-col gap-2 bg-muted/20 rounded-lg p-3 border border-border/30"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="flex items-center gap-2 text-xs font-semibold text-primary mb-1">
                        <Activity className="w-3 h-3 animate-pulse" />
                        <span>AI Agent Processing (Superior to Cursor AI & Manus AI)</span>
                      </div>
                      
                      {(() => {
                        const steps = processStepsObject[activeSessionId] || [];
                        console.log(`🎨 Rendering superior process steps for ${activeSessionId}:`, steps.length, 'steps', processStepsObject);
                        return steps.slice(-5).map((step, index) => ( // Show last 5 steps
                        <motion.div 
                          key={step.id}
                          className={`flex items-center gap-3 text-xs py-1.5 px-2 rounded-md ${
                            step.status === 'completed' ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300' :
                            step.status === 'failed' ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300' :
                            step.status === 'in_progress' ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300' : 
                            'bg-muted/50 text-muted-foreground'
                          }`}
                          initial={{ opacity: 0, x: -10, scale: 0.95 }}
                          animate={{ opacity: 1, x: 0, scale: 1 }}
                          transition={{ 
                            delay: index * 0.1,
                            type: "spring",
                            stiffness: 300,
                            damping: 20
                          }}
                        >
                          {/* Enhanced Status Icons */}
                          <div className="flex-shrink-0">
                            {step.status === 'completed' && (
                              <motion.div
                                initial={{ scale: 0, rotate: -180 }}
                                animate={{ scale: 1, rotate: 0 }}
                                transition={{ type: "spring", stiffness: 400 }}
                              >
                                <CheckCircle className="w-3 h-3 text-green-500" />
                              </motion.div>
                            )}
                            {step.status === 'failed' && (
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: "spring", stiffness: 400 }}
                              >
                                <XCircle className="w-3 h-3 text-red-500" />
                              </motion.div>
                            )}
                            {step.status === 'in_progress' && (
                              <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
                            )}
                            {step.status === 'pending' && (
                              <Clock className="w-3 h-3 text-muted-foreground" />
                            )}
                          </div>
                          
                          {/* Step Description with Type Badge */}
                          <div className="flex-1 flex items-center gap-2">
                            <span className="font-medium">
                              {step.step_description || step.description}
                            </span>
                            
                            {/* Step Type Badge */}
                            {step.step_type && (
                              <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                                step.step_type.includes('ai') ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400' :
                                step.step_type.includes('search') ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400' :
                                step.step_type.includes('automation') ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' :
                                step.step_type.includes('enterprise') ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400' :
                                'bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400'
                              }`}>
                                {step.step_type}
                              </span>
                            )}
                          </div>
                          
                          {/* Timing Information */}
                          {step.timestamp && (
                            <span className="text-xs text-muted-foreground/70 font-mono">
                              {new Date(step.timestamp).toLocaleTimeString()}
                            </span>
                          )}
                        </motion.div>
                      ));
                      })()}
                      
                      {/* Progress indicator for superior tracking */}
                      <div className="mt-2 flex items-center gap-2">
                        <div className="flex-1 bg-muted rounded-full h-1.5">
                          <motion.div 
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-1.5 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ 
                              width: `${Math.min(100, (processStepsObject[activeSessionId] || []).filter((s: any) => s.status === 'completed').length / Math.max(1, (processStepsObject[activeSessionId] || []).length) * 100)}%`
                            }}
                            transition={{ duration: 0.5, ease: "easeOut" }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground font-mono">
                          {(processStepsObject[activeSessionId] || []).filter((s: any) => s.status === 'completed').length}/
                          {(processStepsObject[activeSessionId] || []).length}
                        </span>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {isTyping && !currentProcessSteps.get(activeSessionId)?.length && (
                <motion.div 
                  className="flex items-center gap-2 text-sm text-gray-500"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="flex gap-1">
                    <motion.div 
                      className="w-2 h-2 bg-blue-500 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                    />
                    <motion.div 
                      className="w-2 h-2 bg-blue-500 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                    />
                    <motion.div 
                      className="w-2 h-2 bg-blue-500 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                    />
                  </div>
                  AI is thinking...
                </motion.div>
              )}
            </div>

            <div className="flex items-center gap-2">
              {/* Truncate button - shows when AI is processing */}
              {isTyping && (
                <motion.button
                  onClick={truncateCurrentSession}
                  className="p-2 rounded-lg bg-yellow-100 text-yellow-700 hover:bg-yellow-200 dark:bg-yellow-900 dark:text-yellow-300"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  title="Stop AI processing"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                >
                  <StopCircle size={20} />
                </motion.button>
              )}
              
              <motion.button
                onClick={() => setFullscreenMode(!fullscreenMode)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {fullscreenMode ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
              </motion.button>
              
              <motion.button
                onClick={() => setIsRecording(!isRecording)}
                className={`p-2 rounded-lg ${isRecording ? 'bg-red-100 text-red-600' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {isRecording ? <StopCircle size={20} /> : <PlayCircle size={20} />}
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          <AnimatePresence>
            {activeSession.messages.map((message) => renderEnhancedMessage(message))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <motion.div 
          className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-6"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-end gap-4">
            <div className="flex-1 relative">
              <motion.input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendEnhancedMessage()}
                placeholder="Ask me anything... I'm your agentic AI assistant 🧠✨"
                className="w-full p-4 pr-12 border border-gray-300 dark:border-gray-600 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 resize-none"
                whileFocus={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
              />
              
              <motion.button
                onClick={sendEnhancedMessage}
                disabled={!inputMessage.trim() || !isConnected}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Send size={20} />
              </motion.button>
            </div>

            <div className="flex gap-2">
              <motion.button
                className="p-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Mic size={20} />
              </motion.button>
              
              <motion.button
                className="p-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Camera size={20} />
              </motion.button>
            </div>
          </div>

          {/* Quick Actions */}
          <motion.div 
            className="mt-4 flex flex-wrap gap-2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            {[
              { icon: Search, text: "Search the web", action: () => setInputMessage("Search for ") },
              { icon: Code, text: "Automate task", action: () => setInputMessage("Automate: ") },
              { icon: Camera, text: "Take screenshot", action: () => setInputMessage("Take a screenshot of ") },
              { icon: Brain, text: "Analyze data", action: () => setInputMessage("Analyze this data: ") }
            ].map((quickAction, index) => (
              <motion.button
                key={index}
                onClick={quickAction.action}
                className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
              >
                <quickAction.icon size={16} />
                {quickAction.text}
              </motion.button>
            ))}
          </motion.div>
        </motion.div>
      </div>
      
      {/* Floating Status Button (when popup is hidden) */}
      <AnimatePresence>
        {!statusPopupVisible && (
          (processStepsObject[activeSessionId] && processStepsObject[activeSessionId].length > 0) || 
          activeProcesses.size > 0 || 
          typingStatus.length > 0 || 
          connectionStatus !== 'connected'
        ) && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8, x: 100 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            exit={{ opacity: 0, scale: 0.8, x: 100 }}
            onClick={() => setStatusPopupVisible(true)}
            className="fixed bottom-4 right-4 z-40 bg-blue-500 hover:bg-blue-600 text-white rounded-full p-3 shadow-lg hover:shadow-xl transition-all duration-200"
            title="Show System Status"
          >
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-400' :
                connectionStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' :
                connectionStatus === 'error' ? 'bg-red-400' : 'bg-gray-400'
              }`} />
              {(activeProcesses.size > 0 || typingStatus.length > 0) && (
                <Loader2 size={16} className="animate-spin" />
              )}
              {(activeProcesses.size === 0 && typingStatus.length === 0) && (
                <Activity size={16} />
              )}
            </div>
          </motion.button>
        )}
      </AnimatePresence>
      
      {/* Bottom Right Status Popup */}
      <AnimatePresence>
        {statusPopupVisible && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, x: 100, y: 100 }}
            animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, x: 100, y: 100 }}
            className="fixed bottom-4 right-4 z-50"
          >
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-200 dark:border-gray-700 min-w-[300px] max-w-[400px]">
              {/* Header */}
              <div className="flex items-center justify-between p-3 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500' :
                    connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                    connectionStatus === 'error' ? 'bg-red-500' : 'bg-gray-400'
                  }`} />
                  <span className="font-medium text-sm text-gray-800 dark:text-gray-200">
                    System Status
                  </span>
                </div>
                
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setStatusPopupMinimized(!statusPopupMinimized)}
                    className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                  >
                    {statusPopupMinimized ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  <button
                    onClick={() => setStatusPopupVisible(false)}
                    className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>
              
              {/* Content */}
              <AnimatePresence>
                {!statusPopupMinimized && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="p-3 space-y-3">
                      {/* Connection Status */}
                      <div className="flex items-center gap-2">
                        <Globe size={16} className="text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-300">
                          {connectionStatus === 'connected' ? 'Connected to AI Backend' :
                           connectionStatus === 'connecting' ? 'Connecting to Backend...' :
                           connectionStatus === 'error' ? 'Connection Error' : 'Disconnected'}
                        </span>
                      </div>
                      
                      {/* Real-Time AI Process Steps */}
                      {processStepsObject[activeSessionId] && processStepsObject[activeSessionId].length > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Brain size={16} className="text-purple-500" />
                            <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                              AI Agent Processing (Superior to Cursor AI & Manus AI)
                            </span>
                          </div>
                          <div className="space-y-1 max-h-40 overflow-y-auto">
                            {(processStepsObject[activeSessionId] || [])
                              .slice(-5) // Show last 5 steps
                              .map((step, index) => {
                                const isCompleted = step.status === 'completed';
                                const isInProgress = step.status === 'in_progress';
                                const isError = step.status === 'error' || step.status === 'failed';
                                
                                return (
                                  <div key={step.id || index} className="flex items-start gap-2 text-xs pl-4">
                                    {isCompleted && <div className="w-2 h-2 bg-green-500 rounded-full mt-1 flex-shrink-0"></div>}
                                    {isInProgress && <Loader2 size={12} className="animate-spin text-blue-500 mt-0.5 flex-shrink-0" />}
                                    {isError && <div className="w-2 h-2 bg-red-500 rounded-full mt-1 flex-shrink-0"></div>}
                                    <div className={`${
                                      isCompleted ? 'text-green-700 dark:text-green-400' :
                                      isInProgress ? 'text-blue-700 dark:text-blue-400' :
                                      isError ? 'text-red-700 dark:text-red-400' :
                                      'text-gray-600 dark:text-gray-400'
                                    }`}>
                                      <div className="font-medium">
                                        {step.description || step.step_description || step.step_type || 'Processing...'}
                                      </div>
                                      {step.details && Object.keys(step.details).length > 0 && (
                                        <div className="text-xs opacity-75 mt-0.5">
                                          {typeof step.details === 'string' 
                                            ? step.details 
                                            : Object.entries(step.details).map(([key, value]) => 
                                                `${key}: ${value}`
                                              ).join(', ')
                                          }
                                        </div>
                                      )}
                                      <div className="text-xs opacity-60 mt-0.5">
                                        {new Date(step.timestamp || Date.now()).toLocaleTimeString()}
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      )}
                      
                      {/* Fallback: Show typing status if no detailed steps available */}
                      {(!processStepsObject[activeSessionId] || processStepsObject[activeSessionId].length === 0) && typingStatus.length > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Bot size={16} className="text-green-500" />
                            <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                              AI Processing
                            </span>
                          </div>
                          {typingStatus.map((status, index) => (
                            <div key={index} className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 pl-4">
                              <Loader2 size={12} className="animate-spin" />
                              {status.message}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Toast Notification */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: 50, x: 50 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, y: 50, x: 50 }}
            className="fixed bottom-20 right-4 z-50"
          >
            <div className={`
              px-4 py-3 rounded-lg shadow-lg border max-w-sm
              ${toastType === 'success' ? 'bg-green-50 border-green-200 text-green-800' : ''}
              ${toastType === 'warning' ? 'bg-yellow-50 border-yellow-200 text-yellow-800' : ''}
              ${toastType === 'error' ? 'bg-red-50 border-red-200 text-red-800' : ''}
              ${toastType === 'info' ? 'bg-blue-50 border-blue-200 text-blue-800' : ''}
            `}>
              <div className="flex items-center gap-2">
                {toastType === 'success' && <div className="w-2 h-2 bg-green-500 rounded-full"></div>}
                {toastType === 'warning' && <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>}
                {toastType === 'error' && <div className="w-2 h-2 bg-red-500 rounded-full"></div>}
                {toastType === 'info' && <div className="w-2 h-2 bg-blue-500 rounded-full"></div>}
                <span className="text-sm font-medium">{toastMessage}</span>
                <button 
                  onClick={() => setToastMessage(null)}
                  className="ml-2 text-gray-500 hover:text-gray-700"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
