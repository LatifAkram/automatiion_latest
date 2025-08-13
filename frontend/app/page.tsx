'use client';

import React from 'react';
import { Toaster } from 'react-hot-toast';
import EnhancedChatInterface from '@/components/enhanced-chat-interface';

export default function HomePage() {
  const handleSendMessage = (message: string, metadata?: any) => {
    console.log('Sending message:', message, metadata);
    // Here you would integrate with your backend API
  };

  const handleUserInput = (messageId: string, inputData: any) => {
    console.log('User input received:', messageId, inputData);
    // Handle user input and continue automation
  };

  const handleAutomationControl = (action: 'play' | 'pause' | 'stop' | 'resume', automationId: string) => {
    console.log('Automation control:', action, automationId);
    // Control automation execution
  };

  const handleExport = (format: 'excel' | 'pdf' | 'word', data: any) => {
    console.log('Exporting:', format, data);
    // Handle file export
  };

  const handleCopyToClipboard = (text: string) => {
    console.log('Copied to clipboard:', text);
    // Track clipboard usage
  };

  const handleShareMessage = (messageId: string) => {
    console.log('Sharing message:', messageId);
    // Handle message sharing
  };

  return (
    <div className="h-screen">
      <EnhancedChatInterface
        onSendMessage={handleSendMessage}
        onUserInput={handleUserInput}
        onAutomationControl={handleAutomationControl}
        onExport={handleExport}
        onCopyToClipboard={handleCopyToClipboard}
        onShareMessage={handleShareMessage}
      />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10B981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#EF4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </div>
  );
}