'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  FileSpreadsheet,
  FilePdf,
  FileWord,
  Download,
  Settings,
  CheckCircle,
  AlertCircle,
  Clock,
  X,
  Eye,
  Share2,
  Copy,
  Printer,
  Mail
} from 'lucide-react';
import toast from 'react-hot-toast';

interface ExportData {
  title: string;
  content: any;
  metadata?: {
    author?: string;
    date?: string;
    version?: string;
    tags?: string[];
  };
}

interface ExportOptions {
  format: 'excel' | 'pdf' | 'word';
  includeScreenshots: boolean;
  includeCode: boolean;
  includeLogs: boolean;
  customStyling: boolean;
  pageSize?: 'A4' | 'Letter' | 'Legal';
  orientation?: 'portrait' | 'landscape';
  margins?: {
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
}

interface ResultExporterProps {
  data: ExportData;
  onExport: (options: ExportOptions) => Promise<void>;
  onPreview: (format: 'excel' | 'pdf' | 'word') => void;
  onShare: (format: 'excel' | 'pdf' | 'word', url: string) => void;
}

export default function ResultExporter({
  data,
  onExport,
  onPreview,
  onShare
}: ResultExporterProps) {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'pdf',
    includeScreenshots: true,
    includeCode: true,
    includeLogs: true,
    customStyling: true,
    pageSize: 'A4',
    orientation: 'portrait',
    margins: {
      top: 1,
      bottom: 1,
      left: 1,
      right: 1
    }
  });

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      // Simulate export progress
      const progressInterval = setInterval(() => {
        setExportProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      await onExport(exportOptions);
      
      clearInterval(progressInterval);
      setExportProgress(100);
      
      setTimeout(() => {
        setIsExporting(false);
        setExportProgress(0);
        toast.success(`${exportOptions.format.toUpperCase()} export completed successfully!`);
      }, 1000);

    } catch (error) {
      setIsExporting(false);
      setExportProgress(0);
      toast.error(`Export failed: ${error}`);
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'excel': return <FileSpreadsheet className="w-5 h-5" />;
      case 'pdf': return <FilePdf className="w-5 h-5" />;
      case 'word': return <FileWord className="w-5 h-5" />;
      default: return <FilePdf className="w-5 h-5" />;
    }
  };

  const getFormatColor = (format: string) => {
    switch (format) {
      case 'excel': return 'text-green-600 bg-green-100';
      case 'pdf': return 'text-red-600 bg-red-100';
      case 'word': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Export Results</h2>
          <p className="text-sm text-gray-500">Generate professional reports and documents</p>
        </div>
        <button
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          className="p-2 rounded-lg hover:bg-gray-100"
        >
          <Settings className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Export Progress */}
      {isExporting && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200"
        >
          <div className="flex items-center gap-3 mb-3">
            <Clock className="w-5 h-5 text-blue-600 animate-spin" />
            <span className="font-medium text-blue-800">
              Exporting {exportOptions.format.toUpperCase()}...
            </span>
          </div>
          <div className="w-full bg-blue-200 rounded-full h-2">
            <motion.div
              className="bg-blue-600 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${exportProgress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <div className="text-sm text-blue-600 mt-2">
            {exportProgress}% complete
          </div>
        </motion.div>
      )}

      {/* Format Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Choose Export Format</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(['excel', 'pdf', 'word'] as const).map((format) => (
            <motion.div
              key={format}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                exportOptions.format === format
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setExportOptions(prev => ({ ...prev, format }))}
            >
              <div className="flex items-center gap-3 mb-3">
                {getFormatIcon(format)}
                <span className="font-medium capitalize">{format}</span>
                {exportOptions.format === format && (
                  <CheckCircle className="w-5 h-5 text-blue-500" />
                )}
              </div>
              <p className="text-sm text-gray-600">
                {format === 'excel' && 'Spreadsheet with data and charts'}
                {format === 'pdf' && 'Professional document with formatting'}
                {format === 'word' && 'Editable document with rich content'}
              </p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Content Options */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Content Options</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
            <input
              type="checkbox"
              checked={exportOptions.includeScreenshots}
              onChange={(e) => setExportOptions(prev => ({ ...prev, includeScreenshots: e.target.checked }))}
              className="rounded"
            />
            <div>
              <div className="font-medium">Include Screenshots</div>
              <div className="text-sm text-gray-500">Add automation screenshots</div>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
            <input
              type="checkbox"
              checked={exportOptions.includeCode}
              onChange={(e) => setExportOptions(prev => ({ ...prev, includeCode: e.target.checked }))}
              className="rounded"
            />
            <div>
              <div className="font-medium">Include Code</div>
              <div className="text-sm text-gray-500">Add generated code snippets</div>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
            <input
              type="checkbox"
              checked={exportOptions.includeLogs}
              onChange={(e) => setExportOptions(prev => ({ ...prev, includeLogs: e.target.checked }))}
              className="rounded"
            />
            <div>
              <div className="font-medium">Include Logs</div>
              <div className="text-sm text-gray-500">Add execution logs</div>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
            <input
              type="checkbox"
              checked={exportOptions.customStyling}
              onChange={(e) => setExportOptions(prev => ({ ...prev, customStyling: e.target.checked }))}
              className="rounded"
            />
            <div>
              <div className="font-medium">Custom Styling</div>
              <div className="text-sm text-gray-500">Apply professional formatting</div>
            </div>
          </label>
        </div>
      </div>

      {/* Advanced Options */}
      {showAdvancedOptions && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-6"
        >
          <h3 className="text-lg font-semibold mb-4">Advanced Options</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Page Size</label>
              <select
                value={exportOptions.pageSize}
                onChange={(e) => setExportOptions(prev => ({ ...prev, pageSize: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="A4">A4</option>
                <option value="Letter">Letter</option>
                <option value="Legal">Legal</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Orientation</label>
              <select
                value={exportOptions.orientation}
                onChange={(e) => setExportOptions(prev => ({ ...prev, orientation: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="portrait">Portrait</option>
                <option value="landscape">Landscape</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Top Margin (inches)</label>
              <input
                type="number"
                value={exportOptions.margins?.top}
                onChange={(e) => setExportOptions(prev => ({
                  ...prev,
                  margins: { ...prev.margins!, top: parseFloat(e.target.value) }
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                step="0.1"
                min="0"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Bottom Margin (inches)</label>
              <input
                type="number"
                value={exportOptions.margins?.bottom}
                onChange={(e) => setExportOptions(prev => ({
                  ...prev,
                  margins: { ...prev.margins!, bottom: parseFloat(e.target.value) }
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                step="0.1"
                min="0"
              />
            </div>
          </div>
        </motion.div>
      )}

      {/* Document Preview */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Document Preview</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPreview(exportOptions.format)}
              className="flex items-center gap-2 px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
            >
              <Eye className="w-4 h-4" />
              Preview
            </button>
          </div>
        </div>
        
        <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
          <div className="flex items-center gap-3 mb-3">
            {getFormatIcon(exportOptions.format)}
            <div>
              <div className="font-medium">{data.title}</div>
              <div className="text-sm text-gray-500">
                {exportOptions.format.toUpperCase()} • {data.metadata?.date || new Date().toLocaleDateString()}
              </div>
            </div>
          </div>
          
          <div className="text-sm text-gray-600">
            <div>Author: {data.metadata?.author || 'Automation Platform'}</div>
            <div>Version: {data.metadata?.version || '1.0'}</div>
            {data.metadata?.tags && (
              <div className="flex gap-1 mt-2">
                {data.metadata.tags.map((tag, index) => (
                  <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleExport}
          disabled={isExporting}
          className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          <Download className="w-5 h-5" />
          {isExporting ? 'Exporting...' : `Export ${exportOptions.format.toUpperCase()}`}
        </button>

        <button
          onClick={() => onShare(exportOptions.format, 'https://example.com/share')}
          className="flex items-center gap-2 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
        >
          <Share2 className="w-5 h-5" />
          Share
        </button>

        <button
          onClick={() => window.print()}
          className="flex items-center gap-2 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
        >
          <Printer className="w-5 h-5" />
          Print
        </button>

        <button
          onClick={() => {/* Handle email */}}
          className="flex items-center gap-2 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
        >
          <Mail className="w-5 h-5" />
          Email
        </button>
      </div>

      {/* Export History */}
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-4">Recent Exports</h3>
        <div className="space-y-2">
          {[
            { format: 'pdf', title: 'Automation Report', date: '2024-01-15', size: '2.3 MB' },
            { format: 'excel', title: 'Data Analysis', date: '2024-01-14', size: '1.8 MB' },
            { format: 'word', title: 'Technical Documentation', date: '2024-01-13', size: '3.1 MB' }
          ].map((export_, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${getFormatColor(export_.format)}`}>
                  {getFormatIcon(export_.format)}
                </div>
                <div>
                  <div className="font-medium">{export_.title}</div>
                  <div className="text-sm text-gray-500">{export_.date} • {export_.size}</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-1 rounded hover:bg-gray-200">
                  <Download className="w-4 h-4" />
                </button>
                <button className="p-1 rounded hover:bg-gray-200">
                  <Share2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}