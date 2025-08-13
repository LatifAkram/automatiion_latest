'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Bot, 
  User, 
  Search, 
  Image, 
  Code, 
  Download,
  Play,
  Pause,
  Settings,
  HelpCircle,
  CheckCircle,
  AlertCircle,
  Clock,
  Zap,
  Globe,
  Database,
  Cpu,
  Brain,
  Eye,
  Hand,
  MessageSquare,
  FileSpreadsheet,
  FileText,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Copy,
  Share2
} from 'lucide-react';

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

interface SimpleChatInterfaceProps {
  onSendMessage: (message: string) => void;
  onAutomationControl: (action: string, messageId: string) => void;
  onUserInput: (messageId: string, data: any) => void;
  onCopyToClipboard: (text: string) => void;
  messages: Message[];
  isTyping: boolean;
  activeAutomation: string | null;
}

export default function SimpleChatInterface({
  onSendMessage,
  onAutomationControl,
  onUserInput,
  onCopyToClipboard,
  messages,
  isTyping,
  activeAutomation
}: SimpleChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (inputMessage.trim()) {
      onSendMessage(inputMessage);
      setInputMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    onCopyToClipboard(text);
  };

  const renderMessage = (message: Message) => (
    <motion.div
      key={message.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 p-4 ${
        message.type === 'user' ? 'justify-end' : 'justify-start'
      }`}
    >
      {message.type === 'ai' && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
            <Bot className="w-4 h-4 text-blue-600" />
          </div>
        </div>
      )}
      
      <div className={`max-w-[70%] ${
        message.type === 'user' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
      } rounded-lg p-3 shadow-sm`}>
        <div className="flex items-start justify-between gap-2">
          <p className={`text-sm whitespace-pre-wrap ${
            message.type === 'user' ? 'text-white' : 'text-gray-900 dark:text-gray-100'
          }`}>{message.content}</p>
          <span className={`text-xs ${
            message.type === 'user' ? 'text-white opacity-70' : 'text-gray-500 dark:text-gray-400'
          }`}>
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
        
        {message.automation && (
          <div className="mt-3 p-2 bg-gray-50 dark:bg-gray-700 rounded">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-orange-500" />
              <span className="text-xs font-medium text-gray-900 dark:text-gray-100">Automation: {message.automation.type}</span>
              {message.automation.status === 'handoff_required' && (
                <span className="text-xs bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 px-2 py-1 rounded">Handoff Required</span>
              )}
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${
                  message.automation.status === 'completed' ? 'bg-green-500' :
                  message.automation.status === 'failed' ? 'bg-red-500' :
                  message.automation.status === 'paused' ? 'bg-yellow-500' :
                  message.automation.status === 'handoff_required' ? 'bg-orange-500' :
                  'bg-blue-500'
                }`}
                style={{ width: `${message.automation.progress}%` }}
              />
            </div>
            <div className="flex gap-2 mt-2">
              {message.automation.status === 'running' && (
                <>
                  <button 
                    onClick={() => onAutomationControl('pause', message.id)}
                    className="p-1 rounded bg-yellow-100 dark:bg-yellow-900 hover:bg-yellow-200 dark:hover:bg-yellow-800"
                    title="Pause Automation"
                  >
                    <Pause className="w-3 h-3 text-yellow-600 dark:text-yellow-400" />
                  </button>
                  <button 
                    onClick={() => onAutomationControl('handoff', message.id)}
                    className="p-1 rounded bg-orange-100 dark:bg-orange-900 hover:bg-orange-200 dark:hover:bg-orange-800"
                    title="Request Human Intervention"
                  >
                    <Hand className="w-3 h-3 text-orange-600 dark:text-orange-400" />
                  </button>
                </>
              )}
              {message.automation.status === 'paused' && (
                <button 
                  onClick={() => onAutomationControl('play', message.id)}
                  className="p-1 rounded bg-green-100 dark:bg-green-900 hover:bg-green-200 dark:hover:bg-green-800"
                  title="Resume Automation"
                >
                  <Play className="w-3 h-3 text-green-600 dark:text-green-400" />
                </button>
              )}
              {message.automation.status === 'handoff_required' && (
                <div className="text-xs text-orange-700 dark:text-orange-300">
                  {message.automation.handoffReason}
                </div>
              )}
            </div>
            {message.automation.screenshots && message.automation.screenshots.length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs font-medium mb-2">Screenshots:</h4>
                <div className="grid grid-cols-2 gap-2">
                  {message.automation.screenshots.map((screenshot, index) => (
                    <div key={index} className="text-xs text-gray-600">
                      <div className="bg-gray-200 h-16 rounded flex items-center justify-center">
                        <Image className="w-4 h-4 text-gray-400" />
                      </div>
                      <div className="mt-1 truncate">{screenshot.action}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <h4 className="text-xs font-medium mb-2">Sources:</h4>
            <div className="space-y-2">
              {message.sources.map((source, index) => (
                <div key={index} className="p-2 bg-gray-50 rounded text-xs">
                  <div className="flex items-center gap-2 mb-1">
                    <Globe className="w-3 h-3 text-blue-500" />
                    <a 
                      href={source.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline truncate"
                    >
                      {source.title}
                    </a>
                    <ExternalLink className="w-3 h-3 text-gray-400" />
                  </div>
                  <p className="text-gray-600 line-clamp-2">{source.snippet}</p>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {message.files && message.files.length > 0 && (
          <div className="mt-3">
            <h4 className="text-xs font-medium mb-2">Files:</h4>
            <div className="space-y-2">
              {message.files.map((file, index) => (
                <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                  <FileText className="w-4 h-4 text-gray-500" />
                  <span className="text-xs truncate">{file.name}</span>
                  <span className="text-xs text-gray-500">{file.size}</span>
                  <button 
                    onClick={() => copyToClipboard(file.url)}
                    className="p-1 rounded hover:bg-gray-200"
                  >
                    <Copy className="w-3 h-3 text-gray-400" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {message.type === 'user' && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-gray-600" />
          </div>
        </div>
      )}
    </motion.div>
  );

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
            <Brain className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
              Autonomous Automation Platform
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              AI-powered workflow automation
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map(renderMessage)}
        </AnimatePresence>
        
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3 p-4"
          >
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-blue-600" />
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
              <div className="flex items-center gap-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-gray-500 dark:text-gray-400">AI is thinking...</span>
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="flex gap-3">
          <div className="flex-1">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me to create an automation workflow..."
              className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              rows={1}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isTyping}
            className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}