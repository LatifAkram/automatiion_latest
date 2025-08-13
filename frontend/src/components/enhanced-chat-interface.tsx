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
  Stop,
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
  FileWord,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Copy,
  Share2
} from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import toast from 'react-hot-toast';

interface Source {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  relevance: number;
  source: 'google' | 'bing' | 'github' | 'stackoverflow' | 'duckduckgo';
  preview?: string;
}

interface AutomationStep {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: number;
  result?: any;
  error?: string;
}

interface GeneratedFile {
  name: string;
  type: 'excel' | 'pdf' | 'word' | 'image' | 'code';
  url: string;
  size: string;
  content?: string;
}

interface AutomationStatus {
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  steps: AutomationStep[];
  screenshots?: string[];
  files?: GeneratedFile[];
  performance?: {
    executionTime: number;
    memoryUsage: number;
    cpuUsage: number;
  };
}

interface CodeBlock {
  language: string;
  code: string;
  filename?: string;
  lineNumbers?: boolean;
}

interface UserInputField {
  id: string;
  label: string;
  type: 'text' | 'email' | 'password' | 'select' | 'file' | 'date' | 'number';
  required: boolean;
  options?: string[];
  placeholder?: string;
  validation?: {
    pattern?: string;
    min?: number;
    max?: number;
  };
}

interface Message {
  id: string;
  type: 'user' | 'ai' | 'system' | 'automation' | 'search' | 'error';
  content: string;
  timestamp: Date;
  metadata?: {
    sources?: Source[];
    automation?: AutomationStatus;
    code?: CodeBlock;
    files?: GeneratedFile[];
    charts?: Array<{
      type: 'line' | 'bar' | 'pie' | 'scatter';
      data: any;
      title: string;
    }>;
  };
  isStreaming?: boolean;
  requiresUserInput?: boolean;
  userInputFields?: UserInputField[];
  isExpanded?: boolean;
}

interface EnhancedChatInterfaceProps {
  onSendMessage: (message: string, metadata?: any) => void;
  onUserInput: (messageId: string, inputData: any) => void;
  onAutomationControl: (action: 'play' | 'pause' | 'stop' | 'resume', automationId: string) => void;
  onExport: (format: 'excel' | 'pdf' | 'word', data: any) => void;
  onCopyToClipboard: (text: string) => void;
  onShareMessage: (messageId: string) => void;
}

export default function EnhancedChatInterface({
  onSendMessage,
  onUserInput,
  onAutomationControl,
  onExport,
  onCopyToClipboard,
  onShareMessage
}: EnhancedChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeAutomation, setActiveAutomation] = useState<string | null>(null);
  const [userInputData, setUserInputData] = useState<Record<string, any>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response with automation
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: 'I understand your request. Let me help you with that automation task.',
        timestamp: new Date(),
        metadata: {
          automation: {
            status: 'running',
            progress: 0,
            steps: [
              { name: 'Analyzing request', status: 'running' },
              { name: 'Planning automation', status: 'pending' },
              { name: 'Executing tasks', status: 'pending' },
              { name: 'Generating results', status: 'pending' }
            ],
            performance: {
              executionTime: 0,
              memoryUsage: 0,
              cpuUsage: 0
            }
          },
          sources: [
            {
              title: 'Automation Best Practices',
              url: 'https://example.com/automation-guide',
              snippet: 'Learn about the latest automation techniques and best practices for web automation.',
              domain: 'example.com',
              relevance: 0.95,
              source: 'google'
            }
          ]
        }
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
      setActiveAutomation(aiMessage.id);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleAutomationControl = (action: 'play' | 'pause' | 'stop' | 'resume', messageId: string) => {
    onAutomationControl(action, messageId);
    
    if (action === 'stop') {
      setActiveAutomation(null);
    }
  };

  const handleUserInputSubmit = (messageId: string) => {
    const inputData = userInputData[messageId];
    if (inputData) {
      onUserInput(messageId, inputData);
      setUserInputData(prev => {
        const newData = { ...prev };
        delete newData[messageId];
        return newData;
      });
    }
  };

  const toggleMessageExpansion = (messageId: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, isExpanded: !msg.isExpanded } : msg
    ));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
    onCopyToClipboard(text);
  };

  const renderSourceCard = (source: Source, index: number) => (
    <motion.div
      key={index}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
            <Globe className="w-4 h-4 text-blue-600" />
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <a 
              href={source.url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm font-medium text-blue-600 hover:underline truncate"
            >
              {source.title}
            </a>
            <ExternalLink className="w-3 h-3 text-gray-400" />
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {source.source}
            </span>
          </div>
          <p className="text-xs text-gray-500 mb-2">{source.domain}</p>
          <p className="text-sm text-gray-700 line-clamp-2">{source.snippet}</p>
          <div className="flex items-center justify-between mt-2">
            <div className="flex items-center gap-2">
              <div className="w-16 h-1 bg-gray-200 rounded-full">
                <div 
                  className="h-1 bg-green-500 rounded-full"
                  style={{ width: `${source.relevance * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {Math.round(source.relevance * 100)}% relevant
              </span>
            </div>
            <button
              onClick={() => copyToClipboard(source.url)}
              className="p-1 rounded hover:bg-gray-100"
            >
              <Copy className="w-3 h-3 text-gray-400" />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );

  const renderAutomationStatus = (automation: AutomationStatus, messageId: string) => (
    <div className="mt-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-green-500" />
          <span className="text-sm font-medium">Automation Running</span>
          <span className={`px-2 py-1 rounded-full text-xs ${
            automation.status === 'running' ? 'bg-green-100 text-green-700' :
            automation.status === 'completed' ? 'bg-blue-100 text-blue-700' :
            automation.status === 'failed' ? 'bg-red-100 text-red-700' :
            'bg-yellow-100 text-yellow-700'
          }`}>
            {automation.status}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {automation.status === 'running' && (
            <>
              <button
                onClick={() => handleAutomationControl('pause', messageId)}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                title="Pause"
              >
                <Pause className="w-4 h-4 text-blue-600" />
              </button>
              <button
                onClick={() => handleAutomationControl('stop', messageId)}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                title="Stop"
              >
                <Stop className="w-4 h-4 text-red-600" />
              </button>
            </>
          )}
          {automation.status === 'paused' && (
            <button
              onClick={() => handleAutomationControl('resume', messageId)}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="Resume"
            >
              <Play className="w-4 h-4 text-green-600" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2">
        <motion.div 
          className="bg-green-500 h-2 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${automation.progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {automation.steps.map((step, index) => (
          <div key={index} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50">
            {step.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
            {step.status === 'running' && <Clock className="w-4 h-4 text-blue-500 animate-spin" />}
            {step.status === 'pending' && <AlertCircle className="w-4 h-4 text-gray-400" />}
            {step.status === 'failed' && <AlertCircle className="w-4 h-4 text-red-500" />}
            
            <div className="flex-1">
              <span className={`text-sm ${
                step.status === 'completed' ? 'text-green-600' : 
                step.status === 'running' ? 'text-blue-600' : 
                step.status === 'failed' ? 'text-red-600' : 'text-gray-500'
              }`}>
                {step.name}
              </span>
              {step.duration && (
                <span className="text-xs text-gray-400 ml-2">({step.duration}s)</span>
              )}
            </div>
            
            {step.error && (
              <button
                onClick={() => toast.error(step.error)}
                className="text-xs text-red-500 hover:underline"
              >
                View Error
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Performance Metrics */}
      {automation.performance && (
        <div className="grid grid-cols-3 gap-4 p-3 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">Execution Time</div>
            <div className="text-lg font-bold text-blue-600">
              {automation.performance.executionTime.toFixed(1)}s
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">Memory</div>
            <div className="text-lg font-bold text-green-600">
              {automation.performance.memoryUsage.toFixed(1)}MB
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">CPU</div>
            <div className="text-lg font-bold text-purple-600">
              {automation.performance.cpuUsage.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Screenshots */}
      {automation.screenshots && automation.screenshots.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium mb-3">Screenshots</h4>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {automation.screenshots.map((screenshot, index) => (
              <div key={index} className="flex-shrink-0">
                <img
                  src={screenshot}
                  alt={`Screenshot ${index + 1}`}
                  className="w-24 h-16 object-cover rounded border cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => window.open(screenshot, '_blank')}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generated Files */}
      {automation.files && automation.files.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium mb-3">Generated Files</h4>
          <div className="space-y-2">
            {automation.files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  {file.type === 'excel' && <FileSpreadsheet className="w-5 h-5 text-green-600" />}
                  {file.type === 'pdf' && <FileText className="w-5 h-5 text-red-600" />}
                  {file.type === 'word' && <FileWord className="w-5 h-5 text-blue-600" />}
                  {file.type === 'image' && <Image className="w-5 h-5 text-purple-600" />}
                  {file.type === 'code' && <Code className="w-5 h-5 text-orange-600" />}
                  
                  <div>
                    <div className="text-sm font-medium">{file.name}</div>
                    <div className="text-xs text-gray-500">{file.size}</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => onExport(file.type, file)}
                    className="p-2 rounded-lg hover:bg-gray-200 transition-colors"
                    title="Download"
                  >
                    <Download className="w-4 h-4 text-gray-600" />
                  </button>
                  <button
                    onClick={() => copyToClipboard(file.url)}
                    className="p-2 rounded-lg hover:bg-gray-200 transition-colors"
                    title="Copy URL"
                  >
                    <Copy className="w-4 h-4 text-gray-600" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderCodeBlock = (codeBlock: CodeBlock) => (
    <div className="mt-4">
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
          <div className="flex items-center gap-2">
            <Code className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-300">{codeBlock.language}</span>
            {codeBlock.filename && (
              <span className="text-xs text-gray-500">({codeBlock.filename})</span>
            )}
          </div>
          <button
            onClick={() => copyToClipboard(codeBlock.code)}
            className="p-1 rounded hover:bg-gray-700 transition-colors"
            title="Copy code"
          >
            <Copy className="w-4 h-4 text-gray-400" />
          </button>
        </div>
        <SyntaxHighlighter
          language={codeBlock.language}
          style={tomorrow}
          showLineNumbers={codeBlock.lineNumbers}
          customStyle={{ margin: 0, borderRadius: 0 }}
        >
          {codeBlock.code}
        </SyntaxHighlighter>
      </div>
    </div>
  );

  const renderUserInputForm = (message: Message) => {
    if (!message.requiresUserInput || !message.userInputFields) return null;

    return (
      <div className="mt-4">
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2 mb-3">
            <Hand className="w-5 h-5 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">Action Required</span>
          </div>
          <p className="text-sm text-blue-700 mb-4">
            Please provide the following information to continue the automation:
          </p>
          
          <div className="space-y-4">
            {message.userInputFields.map((field) => (
              <div key={field.id}>
                <label className="block text-sm font-medium text-blue-800 mb-2">
                  {field.label}
                  {field.required && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {field.type === 'text' || field.type === 'email' || field.type === 'password' || field.type === 'number' ? (
                  <input
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full px-3 py-2 border border-blue-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={userInputData[message.id]?.[field.id] || ''}
                    onChange={(e) => setUserInputData(prev => ({
                      ...prev,
                      [message.id]: { ...prev[message.id], [field.id]: e.target.value }
                    }))}
                  />
                ) : field.type === 'select' ? (
                  <select 
                    className="w-full px-3 py-2 border border-blue-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={userInputData[message.id]?.[field.id] || ''}
                    onChange={(e) => setUserInputData(prev => ({
                      ...prev,
                      [message.id]: { ...prev[message.id], [field.id]: e.target.value }
                    }))}
                  >
                    <option value="">Select an option</option>
                    {field.options?.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                ) : field.type === 'file' ? (
                  <input
                    type="file"
                    className="w-full px-3 py-2 border border-blue-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    onChange={(e) => setUserInputData(prev => ({
                      ...prev,
                      [message.id]: { ...prev[message.id], [field.id]: e.target.files?.[0] }
                    }))}
                  />
                ) : field.type === 'date' ? (
                  <input
                    type="date"
                    className="w-full px-3 py-2 border border-blue-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={userInputData[message.id]?.[field.id] || ''}
                    onChange={(e) => setUserInputData(prev => ({
                      ...prev,
                      [message.id]: { ...prev[message.id], [field.id]: e.target.value }
                    }))}
                  />
                ) : null}
              </div>
            ))}
            
            <button
              onClick={() => handleUserInputSubmit(message.id)}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-medium"
            >
              Continue Automation
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user';
    const isAI = message.type === 'ai';
    const isAutomation = message.type === 'automation';

    return (
      <motion.div
        key={message.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}
      >
        <div className={`flex max-w-4xl ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start gap-3`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
            isUser ? 'bg-blue-500' : 'bg-green-500'
          }`}>
            {isUser ? <User className="w-6 h-6 text-white" /> : <Bot className="w-6 h-6 text-white" />}
          </div>

          {/* Message Content */}
          <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block p-4 rounded-2xl ${
              isUser 
                ? 'bg-blue-500 text-white' 
                : 'bg-white border border-gray-200 shadow-sm'
            }`}>
              {/* Message Text */}
              <div className="prose prose-sm max-w-none">
                {message.content}
              </div>

              {/* Automation Status */}
              {message.metadata?.automation && renderAutomationStatus(message.metadata.automation, message.id)}

              {/* Search Results */}
              {message.metadata?.sources && (
                <div className="mt-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium">Sources</h4>
                    <button
                      onClick={() => toggleMessageExpansion(message.id)}
                      className="p-1 rounded hover:bg-gray-100"
                    >
                      {message.isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>
                  
                  <AnimatePresence>
                    {message.isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="space-y-3"
                      >
                        {message.metadata.sources.map((source, index) => renderSourceCard(source, index))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {/* Code Blocks */}
              {message.metadata?.code && renderCodeBlock(message.metadata.code)}

              {/* User Input Form */}
              {renderUserInputForm(message)}

              {/* Message Actions */}
              <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-100">
                <button
                  onClick={() => copyToClipboard(message.content)}
                  className="p-1 rounded hover:bg-gray-100 transition-colors"
                  title="Copy message"
                >
                  <Copy className="w-3 h-3 text-gray-400" />
                </button>
                <button
                  onClick={() => onShareMessage(message.id)}
                  className="p-1 rounded hover:bg-gray-100 transition-colors"
                  title="Share message"
                >
                  <Share2 className="w-3 h-3 text-gray-400" />
                </button>
              </div>
            </div>

            {/* Timestamp */}
            <div className={`text-xs text-gray-500 mt-2 ${isUser ? 'text-right' : 'text-left'}`}>
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Autonomous Automation Platform</h1>
              <p className="text-sm text-gray-500">AI-powered automation with real-time execution</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 rounded-lg hover:bg-gray-100">
              <Settings className="w-5 h-5 text-gray-600" />
            </button>
            <button className="p-2 rounded-lg hover:bg-gray-100">
              <HelpCircle className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        <AnimatePresence>
          {messages.map(renderMessage)}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start mb-4"
          >
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl p-4">
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                  <span className="text-sm text-gray-500">AI is thinking...</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe your automation task... (e.g., 'Book a flight from NYC to London for next week')"
              className="w-full px-4 py-3 border border-gray-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim()}
            className="p-3 bg-blue-500 text-white rounded-2xl hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}