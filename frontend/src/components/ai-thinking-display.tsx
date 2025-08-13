'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Search, 
  Code, 
  Globe, 
  Database, 
  Cpu, 
  Eye, 
  MousePointer, 
  Keyboard,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader,
  Sparkles,
  Lightbulb,
  Target,
  ArrowRight,
  Play,
  Pause
} from 'lucide-react';

interface AIThought {
  id: string;
  type: 'analysis' | 'planning' | 'execution' | 'reasoning' | 'decision' | 'observation';
  content: string;
  timestamp: Date;
  status: 'thinking' | 'processing' | 'completed' | 'failed';
  icon?: React.ReactNode;
  color?: string;
}

interface AIThinkingDisplayProps {
  isVisible: boolean;
  onClose: () => void;
  thoughts: AIThought[];
  currentOperation: string;
  isPaused: boolean;
  onPause: () => void;
  onResume: () => void;
}

export default function AIThinkingDisplay({
  isVisible,
  onClose,
  thoughts,
  currentOperation,
  isPaused,
  onPause,
  onResume
}: AIThinkingDisplayProps) {
  const [showDetails, setShowDetails] = useState(false);

  const getThoughtIcon = (type: string) => {
    switch (type) {
      case 'analysis': return <Search className="w-4 h-4" />;
      case 'planning': return <Target className="w-4 h-4" />;
      case 'execution': return <Zap className="w-4 h-4" />;
      case 'reasoning': return <Brain className="w-4 h-4" />;
      case 'decision': return <Lightbulb className="w-4 h-4" />;
      case 'observation': return <Eye className="w-4 h-4" />;
      default: return <Sparkles className="w-4 h-4" />;
    }
  };

  const getThoughtColor = (type: string) => {
    switch (type) {
      case 'analysis': return 'text-blue-500 bg-blue-100 dark:bg-blue-900';
      case 'planning': return 'text-purple-500 bg-purple-100 dark:bg-purple-900';
      case 'execution': return 'text-green-500 bg-green-100 dark:bg-green-900';
      case 'reasoning': return 'text-orange-500 bg-orange-100 dark:bg-orange-900';
      case 'decision': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900';
      case 'observation': return 'text-indigo-500 bg-indigo-100 dark:bg-indigo-900';
      default: return 'text-gray-500 bg-gray-100 dark:bg-gray-700';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'thinking': return <Loader className="w-4 h-4 animate-spin" />;
      case 'processing': return <Play className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <AlertCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        className="fixed top-4 right-4 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-2xl w-96 max-h-[80vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-t-lg">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="font-semibold">AI Thinking</span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={isPaused ? onResume : onPause}
              className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
              title={isPaused ? 'Resume' : 'Pause'}
            >
              {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            </button>
            
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
              title={showDetails ? 'Hide Details' : 'Show Details'}
            >
              <ArrowRight className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-90' : ''}`} />
            </button>
            
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
              title="Close"
            >
              <span className="text-lg">×</span>
            </button>
          </div>
        </div>

        {/* Current Operation */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-900 dark:text-white">Current Operation</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{currentOperation}</p>
        </div>

        {/* Thoughts Stream */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-3">
            {thoughts.filter(thought => thought && thought.id).map((thought, index) => (
              <motion.div
                key={thought.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-3 rounded-lg border transition-all duration-300 ${
                  thought.status === 'completed' 
                    ? 'border-green-200 dark:border-green-700 bg-green-50 dark:bg-green-900/20' 
                    : thought.status === 'processing'
                    ? 'border-blue-200 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20'
                    : thought.status === 'failed'
                    ? 'border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20'
                    : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-full ${getThoughtColor(thought.type)}`}>
                    {getThoughtIcon(thought.type)}
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                        {thought.type}
                      </span>
                      <span className="text-xs text-gray-400 dark:text-gray-500">
                        {thought.timestamp.toLocaleTimeString()}
                      </span>
                      {getStatusIcon(thought.status)}
                    </div>
                    
                    <p className="text-sm text-gray-900 dark:text-white leading-relaxed">
                      {thought.content}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* AI Status Bar */}
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                {thoughts.length} thoughts processed
              </span>
            </div>
            
            <div className="flex items-center gap-1">
              <Brain className="w-3 h-3 text-blue-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">
                AI Active
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}