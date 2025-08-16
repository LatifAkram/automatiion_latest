'use client';

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, Zap, Target, Clock, Shield, Cpu, Eye, Code } from 'lucide-react';

interface SophisticatedData {
  success: boolean;
  aiInterpretation: string;
  aiProvider: string;
  processingPath: string;
  confidence: number;
  processingTime: number;
  fallbackUsed: boolean;
  system: string;
  enhancedParsing?: {
    instruction_type: string;
    intent_category: string;
    complexity_level: string;
    parsing_confidence: number;
    detected_platforms: string[];
    extracted_entities: string[];
    steps_identified: number;
    preprocessing_applied: string[];
    metadata: any;
  };
  detectedComplexity?: string;
  timestamp: string;
  evidence: any[];
  result: any;
}

interface SophisticatedAutomationDisplayProps {
  data: SophisticatedData;
  automationId?: string;
}

export default function SophisticatedAutomationDisplay({ data, automationId }: SophisticatedAutomationDisplayProps) {
  const [expandedSections, setExpandedSections] = useState<{ [key: string]: boolean }>({
    overview: true,
    aiAnalysis: false,
    enhancedParsing: false,
    execution: false,
    evidence: false
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity?.toLowerCase()) {
      case 'simple': return 'text-green-600';
      case 'moderate': return 'text-yellow-600';
      case 'complex': return 'text-orange-600';
      case 'ultra_complex': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const formatProcessingTime = (time: number) => {
    return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(2)}s`;
  };

  return (
    <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-blue-600" />
        <h3 className="font-semibold text-blue-900 dark:text-blue-100">SUPER-OMEGA Sophisticated Analysis - System Overview</h3>
        <span className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 px-2 py-1 rounded-full">
          ID: {automationId?.slice(-8)}
        </span>
      </div>

      {/* Overview Section */}
      <div className="mb-4">
        <button
          onClick={() => toggleSection('overview')}
          className="flex items-center gap-2 w-full text-left font-medium text-gray-800 dark:text-gray-200 hover:text-blue-600 transition-colors"
        >
          {expandedSections.overview ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <Target className="w-4 h-4" />
          System Overview
        </button>
        
        {expandedSections.overview && (
          <div className="ml-6 mt-2 space-y-2">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-600 dark:text-gray-400">System:</span>
                <div className="text-blue-600 font-medium">{data.system}</div>
              </div>
              <div>
                <span className="font-medium text-gray-600 dark:text-gray-400">Processing Path:</span>
                <div className="text-purple-600 font-medium capitalize">{data.processingPath}</div>
              </div>
                              <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">AI Provider:</span>
                  <div className="text-green-600 font-medium capitalize">{data.aiProvider}</div>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">AI Component:</span>
                  <div className="text-purple-600 font-medium capitalize">
                    {data.enhancedParsing?.metadata?.ai_component || 'orchestrator'}
                  </div>
                </div>
              <div>
                <span className="font-medium text-gray-600 dark:text-gray-400">Processing Time:</span>
                <div className="text-orange-600 font-medium">{formatProcessingTime(data.processingTime)}</div>
              </div>
              <div>
                <span className="font-medium text-gray-600 dark:text-gray-400">Confidence:</span>
                <div className={`font-medium ${getConfidenceColor(data.confidence)}`}>
                  {(data.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-600 dark:text-gray-400">Fallback Used:</span>
                <div className={`font-medium ${data.fallbackUsed ? 'text-yellow-600' : 'text-green-600'}`}>
                  {data.fallbackUsed ? 'Yes' : 'No'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* AI Analysis Section */}
      <div className="mb-4">
        <button
          onClick={() => toggleSection('aiAnalysis')}
          className="flex items-center gap-2 w-full text-left font-medium text-gray-800 dark:text-gray-200 hover:text-blue-600 transition-colors"
        >
          {expandedSections.aiAnalysis ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <Brain className="w-4 h-4" />
          AI Interpretation
        </button>
        
        {expandedSections.aiAnalysis && (
          <div className="ml-6 mt-2">
            <div className="bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                {data.aiInterpretation}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Parsing Section */}
      {data.enhancedParsing && (
        <div className="mb-4">
          <button
            onClick={() => toggleSection('enhancedParsing')}
            className="flex items-center gap-2 w-full text-left font-medium text-gray-800 dark:text-gray-200 hover:text-blue-600 transition-colors"
          >
            {expandedSections.enhancedParsing ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            <Code className="w-4 h-4" />
            Enhanced Parsing Analysis
          </button>
          
          {expandedSections.enhancedParsing && (
            <div className="ml-6 mt-2 space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">Instruction Type:</span>
                  <div className="text-blue-600 font-medium capitalize">{data.enhancedParsing.instruction_type}</div>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">Intent Category:</span>
                  <div className="text-purple-600 font-medium capitalize">{data.enhancedParsing.intent_category}</div>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">Complexity Level:</span>
                  <div className={`font-medium ${getComplexityColor(data.enhancedParsing.complexity_level)}`}>
                    {data.enhancedParsing.complexity_level.replace('_', ' ')}
                  </div>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">Parsing Confidence:</span>
                  <div className={`font-medium ${getConfidenceColor(data.enhancedParsing.parsing_confidence)}`}>
                    {(data.enhancedParsing.parsing_confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">Steps Identified:</span>
                  <div className="text-indigo-600 font-medium">{data.enhancedParsing.steps_identified}</div>
                </div>
              </div>
              
              {data.enhancedParsing.detected_platforms.length > 0 && (
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400 block mb-1">Detected Platforms:</span>
                  <div className="flex flex-wrap gap-1">
                    {data.enhancedParsing.detected_platforms.map((platform, index) => (
                      <span key={index} className="text-xs bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-100 px-2 py-1 rounded-full">
                        {platform}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {data.enhancedParsing.preprocessing_applied.length > 0 && (
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400 block mb-1">Preprocessing Applied:</span>
                  <div className="flex flex-wrap gap-1">
                    {data.enhancedParsing.preprocessing_applied.map((process, index) => (
                      <span key={index} className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 px-2 py-1 rounded-full">
                        {process.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Execution Results Section */}
      <div className="mb-4">
        <button
          onClick={() => toggleSection('execution')}
          className="flex items-center gap-2 w-full text-left font-medium text-gray-800 dark:text-gray-200 hover:text-blue-600 transition-colors"
        >
          {expandedSections.execution ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <Zap className="w-4 h-4" />
          Execution Results
        </button>
        
        {expandedSections.execution && (
          <div className="ml-6 mt-2">
            <div className="bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-700">
              <div className="text-sm font-medium mb-2 text-gray-800 dark:text-gray-200">Execution Results Summary:</div>
              <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {JSON.stringify(data.result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Evidence Section */}
      {data.evidence && data.evidence.length > 0 && (
        <div className="mb-4">
          <button
            onClick={() => toggleSection('evidence')}
            className="flex items-center gap-2 w-full text-left font-medium text-gray-800 dark:text-gray-200 hover:text-blue-600 transition-colors"
          >
            {expandedSections.evidence ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            <Eye className="w-4 h-4" />
            Evidence Collection ({data.evidence.length} items)
          </button>
          
          {expandedSections.evidence && (
            <div className="ml-6 mt-2 space-y-2">
              <div className="text-sm font-medium mb-2 text-gray-800 dark:text-gray-200">Evidence Collection Summary:</div>
              {data.evidence.map((item, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-700">
                  <div className="text-sm">
                    <div className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                      {item.type || 'Evidence Item'}
                    </div>
                    <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                      {JSON.stringify(item, null, 2)}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Timestamp */}
      <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1 mt-4">
        <Clock className="w-3 h-3" />
        Processed at {data.timestamp}
      </div>
    </div>
  );
}