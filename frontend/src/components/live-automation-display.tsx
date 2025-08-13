'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Square, 
  Eye, 
  Download, 
  Code, 
  Monitor, 
  MousePointer, 
  Keyboard, 
  Clock,
  CheckCircle,
  AlertCircle,
  Loader,
  Maximize2,
  Minimize2,
  X,
  ChevronDown,
  ChevronUp,
  Zap,
  Globe,
  FileImage,
  Video,
  Settings,
  Share2
} from 'lucide-react';

interface AutomationStep {
  id: string;
  action: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  timestamp: string;
  screenshot?: string;
  duration?: number;
  error?: string;
}

interface LiveAutomationDisplayProps {
  isVisible: boolean;
  onClose: () => void;
  automationId?: string;
  currentStep?: AutomationStep;
  steps: AutomationStep[];
  browserUrl?: string;
  isRunning: boolean;
  onControl: (action: 'play' | 'pause' | 'stop') => void;
  onTakeScreenshot: () => void;
  onViewCode: () => void;
  onDownloadReport: () => void;
}

export default function LiveAutomationDisplay({
  isVisible,
  onClose,
  automationId,
  currentStep,
  steps,
  browserUrl,
  isRunning,
  onControl,
  onTakeScreenshot,
  onViewCode,
  onDownloadReport
}: LiveAutomationDisplayProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [activeTab, setActiveTab] = useState<'browser' | 'steps' | 'screenshots' | 'code'>('browser');
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [selectedScreenshot, setSelectedScreenshot] = useState<string | null>(null);
  const stepsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (stepsEndRef.current) {
      stepsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [steps]);

  const getStepIcon = (action: string) => {
    const actionLower = action.toLowerCase();
    if (actionLower.includes('click')) return <MousePointer className="w-4 h-4" />;
    if (actionLower.includes('type') || actionLower.includes('enter')) return <Keyboard className="w-4 h-4" />;
    if (actionLower.includes('wait') || actionLower.includes('sleep')) return <Clock className="w-4 h-4" />;
    if (actionLower.includes('navigate') || actionLower.includes('open')) return <Globe className="w-4 h-4" />;
    return <Zap className="w-4 h-4" />;
  };

  const getStepStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-500 bg-green-100 dark:bg-green-900';
      case 'running': return 'text-blue-500 bg-blue-100 dark:bg-blue-900 animate-pulse';
      case 'failed': return 'text-red-500 bg-red-100 dark:bg-red-900';
      default: return 'text-gray-500 bg-gray-100 dark:bg-gray-700';
    }
  };

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'running': return <Loader className="w-4 h-4 animate-spin" />;
      case 'failed': return <AlertCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 50 }}
        className={`fixed bottom-4 right-4 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-2xl ${
          isMinimized ? 'w-80 h-16' : showFullscreen ? 'w-full h-full top-0 left-0' : 'w-96 h-96'
        } transition-all duration-300`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded-t-lg">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="font-semibold text-gray-900 dark:text-white">Live Automation</span>
            {automationId && (
              <span className="text-xs text-gray-500 dark:text-gray-400">#{automationId}</span>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            {/* Control Buttons */}
            {!isMinimized && (
              <>
                <button
                  onClick={() => onControl(isRunning ? 'pause' : 'play')}
                  className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
                  title={isRunning ? 'Pause' : 'Play'}
                >
                  {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => onControl('stop')}
                  className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
                  title="Stop"
                >
                  <Square className="w-4 h-4" />
                </button>
                <button
                  onClick={onTakeScreenshot}
                  className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
                  title="Take Screenshot"
                >
                  <FileImage className="w-4 h-4" />
                </button>
              </>
            )}
            
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
              title={isMinimized ? 'Expand' : 'Minimize'}
            >
              {isMinimized ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            <button
              onClick={() => setShowFullscreen(!showFullscreen)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
              title={showFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            >
              {showFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
            
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900 text-red-600 dark:text-red-400"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {!isMinimized && (
          <>
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200 dark:border-gray-700">
              {[
                { id: 'browser', label: 'Browser', icon: <Monitor className="w-4 h-4" /> },
                { id: 'steps', label: 'Steps', icon: <Zap className="w-4 h-4" /> },
                { id: 'screenshots', label: 'Screenshots', icon: <FileImage className="w-4 h-4" /> },
                { id: 'code', label: 'Code', icon: <Code className="w-4 h-4" /> }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden">
              {/* Browser Tab */}
              {activeTab === 'browser' && (
                <div className="h-full flex flex-col">
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <div className="flex-1 bg-white dark:bg-gray-800 rounded px-3 py-1 text-sm text-gray-600 dark:text-gray-400">
                        {browserUrl || 'Loading...'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex-1 bg-gray-100 dark:bg-gray-900 p-4">
                    <div className="w-full h-full bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 flex items-center justify-center">
                      {currentStep ? (
                        <div className="text-center">
                          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
                            {getStepIcon(currentStep.action)}
                          </div>
                          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                            {currentStep.action}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {currentStep.description}
                          </p>
                          <div className="mt-4">
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center text-gray-500 dark:text-gray-400">
                          <Monitor className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p>Browser automation ready</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Steps Tab */}
              {activeTab === 'steps' && (
                <div className="h-full overflow-y-auto p-4">
                  <div className="space-y-3">
                    {steps.map((step, index) => (
                      <motion.div
                        key={`${step.id}_${index}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`p-3 rounded-lg border ${
                          step.status === 'running' 
                            ? 'border-blue-200 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20' 
                            : step.status === 'completed'
                            ? 'border-green-200 dark:border-green-700 bg-green-50 dark:bg-green-900/20'
                            : step.status === 'failed'
                            ? 'border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20'
                            : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-full ${getStepStatusColor(step.status)}`}>
                            {getStepStatusIcon(step.status)}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium text-gray-900 dark:text-white">
                                {step.action}
                              </span>
                              {step.duration && (
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {step.duration}ms
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              {step.description}
                            </p>
                            {step.error && (
                              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                                Error: {step.error}
                              </p>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    ))}
                    <div ref={stepsEndRef} />
                  </div>
                </div>
              )}

              {/* Screenshots Tab */}
              {activeTab === 'screenshots' && (
                <div className="h-full overflow-y-auto p-4">
                  <div className="grid grid-cols-2 gap-3">
                    {steps.filter(step => step.screenshot).map((step, index) => (
                      <motion.div
                        key={`screenshot_${step.id}_${index}`}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="relative group cursor-pointer"
                        onClick={() => setSelectedScreenshot(step.screenshot!)}
                      >
                        <div className="aspect-video bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600">
                          <img
                            src={step.screenshot}
                            alt={`Screenshot: ${step.action}`}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
                          <Eye className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                        </div>
                        <div className="mt-2">
                          <p className="text-xs font-medium text-gray-900 dark:text-white truncate">
                            {step.action}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {step.timestamp}
                          </p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Code Tab */}
              {activeTab === 'code' && (
                <div className="h-full p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-gray-900 dark:text-white">Generated Code</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={onViewCode}
                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        View Full Code
                      </button>
                      <button
                        onClick={onDownloadReport}
                        className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                      >
                        Download Report
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto h-full">
                    <pre>
{`// Playwright Automation Code
import { chromium } from 'playwright';

async function runAutomation() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    // Navigate to target URL
    await page.goto('${browserUrl || 'https://example.com'}');
    
    // Automation steps
${steps.map(step => `    // ${step.action}
    ${step.description}`).join('\n')}
    
    console.log('Automation completed successfully');
  } catch (error) {
    console.error('Automation failed:', error);
  } finally {
    await browser.close();
  }
}

runAutomation();`}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </motion.div>

      {/* Screenshot Modal */}
      <AnimatePresence>
        {selectedScreenshot && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedScreenshot(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-4xl max-h-full"
              onClick={(e) => e.stopPropagation()}
            >
              <img
                src={selectedScreenshot}
                alt="Screenshot"
                className="max-w-full max-h-full object-contain rounded-lg"
              />
              <button
                onClick={() => setSelectedScreenshot(null)}
                className="absolute top-4 right-4 p-2 bg-black bg-opacity-50 text-white rounded-full hover:bg-opacity-75 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </AnimatePresence>
  );
}