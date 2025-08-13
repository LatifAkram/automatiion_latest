"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, Maximize2, Minimize2, ExternalLink, Loader2 } from 'lucide-react';

interface RealTimeBrowserProps {
  url: string;
  isAutomationRunning: boolean;
  currentStep: number;
  totalSteps: number;
  automationSteps: any[];
  screenshots: string[];
  onStepComplete?: (stepIndex: number) => void;
  onAutomationComplete?: () => void;
}

interface BrowserState {
  isLoading: boolean;
  hasError: boolean;
  errorMessage: string;
  currentUrl: string;
  isFullscreen: boolean;
  isPaused: boolean;
}

export const RealTimeBrowser: React.FC<RealTimeBrowserProps> = ({
  url,
  isAutomationRunning,
  currentStep,
  totalSteps,
  automationSteps,
  screenshots,
  onStepComplete,
  onAutomationComplete
}) => {
  const [browserState, setBrowserState] = useState<BrowserState>({
    isLoading: false,
    hasError: false,
    errorMessage: '',
    currentUrl: url,
    isFullscreen: false,
    isPaused: false
  });

  const [currentStepData, setCurrentStepData] = useState<any>(null);
  const [stepProgress, setStepProgress] = useState(0);
  const [automationLog, setAutomationLog] = useState<string[]>([]);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Update current step data when step changes
  useEffect(() => {
    if (automationSteps && automationSteps.length > 0 && currentStep <= automationSteps.length) {
      const step = automationSteps[currentStep - 1];
      setCurrentStepData(step);
      setStepProgress(0);
      
      // Add step to log
      setAutomationLog(prev => [...prev, `Step ${currentStep}: ${step?.action || 'Unknown action'} - ${step?.description || ''}`]);
      
      // Simulate step progress
      if (isAutomationRunning && !browserState.isPaused) {
        progressIntervalRef.current = setInterval(() => {
          setStepProgress(prev => {
            if (prev >= 100) {
              if (onStepComplete) onStepComplete(currentStep - 1);
              return 100;
            }
            return prev + 10;
          });
        }, 500);
      }
    }
  }, [currentStep, automationSteps, isAutomationRunning, browserState.isPaused]);

  // Cleanup interval
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  // Handle automation completion
  useEffect(() => {
    if (currentStep > totalSteps && totalSteps > 0) {
      setAutomationLog(prev => [...prev, '✅ Automation completed successfully!']);
      if (onAutomationComplete) onAutomationComplete();
    }
  }, [currentStep, totalSteps, onAutomationComplete]);

  // Update URL when prop changes
  useEffect(() => {
    setBrowserState(prev => ({ ...prev, currentUrl: url }));
  }, [url]);

  const handleLoadStart = () => {
    setBrowserState(prev => ({ ...prev, isLoading: true, hasError: false }));
  };

  const handleLoadComplete = () => {
    setBrowserState(prev => ({ ...prev, isLoading: false }));
  };

  const handleLoadError = () => {
    setBrowserState(prev => ({ 
      ...prev, 
      isLoading: false, 
      hasError: true, 
      errorMessage: 'Failed to load website' 
    }));
  };

  const toggleFullscreen = () => {
    setBrowserState(prev => ({ ...prev, isFullscreen: !prev.isFullscreen }));
  };

  const togglePause = () => {
    setBrowserState(prev => ({ ...prev, isPaused: !prev.isPaused }));
  };

  const reloadPage = () => {
    if (iframeRef.current) {
      iframeRef.current.src = iframeRef.current.src;
    }
  };

  const openInNewTab = () => {
    window.open(browserState.currentUrl, '_blank');
  };

  const getStepStatus = (stepIndex: number) => {
    if (stepIndex < currentStep - 1) return 'completed';
    if (stepIndex === currentStep - 1) return 'running';
    return 'pending';
  };

  const getStepIcon = (action: string) => {
    switch (action) {
      case 'navigate': return '🌐';
      case 'click': return '👆';
      case 'type': return '⌨️';
      case 'wait': return '⏱️';
      case 'scroll': return '📜';
      default: return '⚡';
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden ${
      browserState.isFullscreen ? 'fixed inset-0 z-50' : 'w-full h-full'
    }`}>
      {/* Browser Header */}
      <div className="bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-3">
        <div className="flex items-center justify-between">
          {/* Traffic Light Buttons */}
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>

          {/* URL Bar */}
          <div className="flex-1 mx-4">
            <div className="bg-white dark:bg-gray-700 rounded-md px-3 py-1 text-sm text-gray-600 dark:text-gray-300 flex items-center">
              {browserState.isLoading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
              <span className="truncate">{browserState.currentUrl}</span>
            </div>
          </div>

          {/* Browser Controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={reloadPage}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
              title="Reload"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={openInNewTab}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
              title="Open in new tab"
            >
              <ExternalLink className="w-4 h-4" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
              title={browserState.isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {browserState.isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Automation Progress Bar */}
      {isAutomationRunning && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-200 dark:border-blue-800 p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
              </div>
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                Automation Running - Step {currentStep} of {totalSteps}
              </span>
            </div>
            <button
              onClick={togglePause}
              className="p-1 hover:bg-blue-100 dark:hover:bg-blue-800 rounded"
              title={browserState.isPaused ? "Resume" : "Pause"}
            >
              {browserState.isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            </button>
          </div>
          
          {/* Overall Progress */}
          <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2 mb-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(currentStep / totalSteps) * 100}%` }}
            ></div>
          </div>

          {/* Current Step Progress */}
          {currentStepData && (
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
              <div 
                className="bg-green-500 h-1 rounded-full transition-all duration-300"
                style={{ width: `${stepProgress}%` }}
              ></div>
            </div>
          )}
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex h-full">
        {/* Browser View */}
        <div className={`flex-1 ${browserState.isFullscreen ? 'h-screen' : 'h-96'}`}>
          {browserState.hasError ? (
            <div className="flex items-center justify-center h-full bg-gray-50 dark:bg-gray-800">
              <div className="text-center">
                <div className="text-red-500 text-6xl mb-4">⚠️</div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                  Failed to Load Website
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {browserState.errorMessage}
                </p>
                <button
                  onClick={reloadPage}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : (
            <div className="relative h-full">
              <iframe
                ref={iframeRef}
                src={browserState.currentUrl}
                className="w-full h-full border-0"
                onLoadStart={handleLoadStart}
                onLoad={handleLoadComplete}
                onError={handleLoadError}
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                title="Real-time browser automation"
              />
              
              {/* Loading Overlay */}
              {browserState.isLoading && (
                <div className="absolute inset-0 bg-white dark:bg-gray-900 bg-opacity-75 flex items-center justify-center">
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-500" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">Loading website...</p>
                  </div>
                </div>
              )}

              {/* Automation Overlay */}
              {isAutomationRunning && currentStepData && (
                <div className="absolute top-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 max-w-sm">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="text-2xl">{getStepIcon(currentStepData.action)}</span>
                    <div>
                      <h4 className="font-semibold text-sm">Step {currentStep}</h4>
                      <p className="text-xs text-gray-600 dark:text-gray-400">{currentStepData.action}</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    {currentStepData.description}
                  </p>
                  {currentStepData.selector && (
                    <div className="bg-gray-100 dark:bg-gray-700 rounded px-2 py-1 text-xs font-mono">
                      {currentStepData.selector}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Sidebar - Steps and Screenshots */}
        {!browserState.isFullscreen && (
          <div className="w-80 border-l border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <div className="p-4">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">Automation Progress</h3>
              
              {/* Steps List */}
              <div className="space-y-2 mb-6">
                {automationSteps.map((step, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-3 rounded-lg border ${
                      getStepStatus(index) === 'completed' 
                        ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' 
                        : getStepStatus(index) === 'running'
                        ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                        : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getStepIcon(step.action)}</span>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          Step {index + 1}: {step.action}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          {step.description}
                        </p>
                      </div>
                      {getStepStatus(index) === 'completed' && (
                        <span className="text-green-500">✅</span>
                      )}
                      {getStepStatus(index) === 'running' && (
                        <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Screenshots */}
              {screenshots.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">Screenshots</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {screenshots.map((screenshot, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="relative group cursor-pointer"
                      >
                        <img
                          src={screenshot}
                          alt={`Screenshot ${index + 1}`}
                          className="w-full h-20 object-cover rounded border border-gray-200 dark:border-gray-600"
                        />
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-all duration-200 rounded flex items-center justify-center">
                          <span className="text-white opacity-0 group-hover:opacity-100 text-xs">
                            Step {index + 1}
                          </span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Automation Log */}
              {automationLog.length > 0 && (
                <div className="mt-6">
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">Log</h4>
                  <div className="bg-white dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600 p-3 max-h-32 overflow-y-auto">
                    {automationLog.map((log, index) => (
                      <div key={index} className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};