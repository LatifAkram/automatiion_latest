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
  Share2,
  ExternalLink
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
  selector?: string;
  value?: string;
}

interface RealBrowserAutomationProps {
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

export default function RealBrowserAutomation({
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
}: RealBrowserAutomationProps) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [activeTab, setActiveTab] = useState<'browser' | 'steps' | 'screenshots' | 'code'>('browser');
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [selectedScreenshot, setSelectedScreenshot] = useState<string | null>(null);
  const [browserView, setBrowserView] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const stepsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (stepsEndRef.current) {
      stepsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [steps]);

  // Simulate real browser automation
  useEffect(() => {
    if (isRunning && browserUrl) {
      setIsLoading(true);
      
      // Simulate browser loading
      setTimeout(() => {
        setBrowserView(`
          <div style="background: white; padding: 20px; font-family: Arial, sans-serif;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
              <h2 style="color: #333; margin: 0 0 10px 0;">🌐 Browser Automation Active</h2>
              <p style="color: #666; margin: 0;">Currently automating: <strong>${browserUrl}</strong></p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
              <h3 style="color: #1976d2; margin: 0 0 10px 0;">Current Action</h3>
              <p style="color: #424242; margin: 0;">${currentStep?.action || 'Initializing automation...'}</p>
              <p style="color: #666; font-size: 14px; margin: 5px 0 0 0;">${currentStep?.description || ''}</p>
            </div>
            
            <div style="background: #f3e5f5; padding: 15px; border-radius: 8px;">
              <h3 style="color: #7b1fa2; margin: 0 0 10px 0;">Automation Progress</h3>
              <div style="background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #4caf50, #8bc34a); height: 100%; width: ${Math.min((steps.filter(s => s.status === 'completed').length / Math.max(steps.length, 1)) * 100, 100)}%; transition: width 0.3s ease;"></div>
              </div>
              <p style="color: #666; font-size: 14px; margin: 10px 0 0 0;">
                ${steps.filter(s => s.status === 'completed').length} of ${steps.length} steps completed
              </p>
            </div>
          </div>
        `);
        setIsLoading(false);
      }, 1000);
    }
  }, [isRunning, browserUrl, currentStep, steps]);

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
          isMinimized ? 'w-80 h-16' : showFullscreen ? 'w-full h-full top-0 left-0' : 'w-[700px] h-[600px]'
        } transition-all duration-300`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-green-500 to-blue-600 text-white rounded-t-lg">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="font-semibold">🚀 Real Browser Automation</span>
            {automationId && (
              <span className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">#{automationId}</span>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            {/* Control Buttons */}
            {!isMinimized && (
              <>
                <button
                  onClick={() => onControl(isRunning ? 'pause' : 'play')}
                  className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
                  title={isRunning ? 'Pause' : 'Play'}
                >
                  {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => onControl('stop')}
                  className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
                  title="Stop"
                >
                  <Square className="w-4 h-4" />
                </button>
                <button
                  onClick={onTakeScreenshot}
                  className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
                  title="Take Screenshot"
                >
                  <FileImage className="w-4 h-4" />
                </button>
              </>
            )}
            
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
              title={isMinimized ? 'Expand' : 'Minimize'}
            >
              {isMinimized ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            <button
              onClick={() => setShowFullscreen(!showFullscreen)}
              className="p-1 rounded hover:bg-white hover:bg-opacity-20 transition-colors"
              title={showFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            >
              {showFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
            
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-red-500 hover:bg-opacity-20 transition-colors"
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
                { id: 'browser', label: 'Live Browser', icon: <Monitor className="w-4 h-4" /> },
                { id: 'steps', label: 'Automation Steps', icon: <Zap className="w-4 h-4" /> },
                { id: 'screenshots', label: 'Screenshots', icon: <FileImage className="w-4 h-4" /> },
                { id: 'code', label: 'Generated Code', icon: <Code className="w-4 h-4" /> }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'text-green-600 dark:text-green-400 border-b-2 border-green-600 dark:border-green-400'
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
                      <div className="flex gap-1">
                        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                        <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      </div>
                      <div className="flex-1 bg-white dark:bg-gray-800 rounded px-3 py-1 text-sm text-gray-600 dark:text-gray-400 mx-2">
                        {browserUrl || 'Loading...'}
                      </div>
                      <ExternalLink className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>
                  
                  <div className="flex-1 bg-gray-100 dark:bg-gray-900 p-4">
                    <div className="w-full h-full bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                      {isLoading ? (
                        <div className="flex items-center justify-center h-full">
                          <div className="text-center">
                            <Loader className="w-8 h-8 animate-spin text-green-500 mx-auto mb-4" />
                            <p className="text-gray-600 dark:text-gray-400">Loading browser automation...</p>
                          </div>
                        </div>
                      ) : browserView ? (
                        <div 
                          className="h-full overflow-auto"
                          dangerouslySetInnerHTML={{ __html: browserView }}
                        />
                      ) : (
                        <div className="text-center text-gray-500 dark:text-gray-400 p-8">
                          <div className="w-24 h-24 bg-gradient-to-r from-green-100 to-blue-100 dark:from-green-900 dark:to-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Globe className="w-12 h-12 text-green-500 dark:text-green-400" />
                          </div>
                          <h3 className="text-lg font-semibold mb-2">Real Browser Automation Ready</h3>
                          <p className="text-sm">Click play to start real browser automation</p>
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
                        key={`real_step_${step.id || `step_${index}`}_${step.action?.replace(/\s+/g, '_') || 'step'}_${Date.now()}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`p-3 rounded-lg border ${
                          step.status === 'running' 
                            ? 'border-green-200 dark:border-green-700 bg-green-50 dark:bg-green-900/20' 
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
                            {step.selector && (
                              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                                Selector: {step.selector}
                              </p>
                            )}
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
                  <div className="grid grid-cols-3 gap-4">
                    {steps.filter(step => step.screenshot).map((step, index) => (
                      <motion.div
                        key={`real_screenshot_${step.id || `screenshot_${index}`}_${step.action?.replace(/\s+/g, '_') || 'screenshot'}_${Date.now()}`}
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
                    <h3 className="font-semibold text-gray-900 dark:text-white">Generated Automation Code</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={onViewCode}
                        className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                      >
                        View Full Code
                      </button>
                      <button
                        onClick={onDownloadReport}
                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        Download Report
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto h-full">
                    <pre>
{`// Real Browser Automation Code
import { chromium } from 'playwright';

async function runRealAutomation() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    // Navigate to target URL
    await page.goto('${browserUrl || 'https://example.com'}');
    
    // Real automation steps
${steps.map(step => `    // ${step.action}
    ${step.description}
    ${step.selector ? `await page.click('${step.selector}');` : ''}
    ${step.value ? `await page.fill('${step.selector}', '${step.value}');` : ''}`).join('\n')}
    
    console.log('Real automation completed successfully');
  } catch (error) {
    console.error('Real automation failed:', error);
  } finally {
    await browser.close();
  }
}

runRealAutomation();`}
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