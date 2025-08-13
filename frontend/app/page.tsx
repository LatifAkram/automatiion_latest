'use client';

import React, { useState, useEffect } from 'react';
import SimpleChatInterface from '../src/components/simple-chat-interface';

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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [activeAutomation, setActiveAutomation] = useState<string | null>(null);

  const handleSendMessage = async (message: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      // Send message to backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: 'default_session',
          context: {
            domain: 'general',
            user_preferences: {
              automation_type: 'general',
              complexity: 'medium'
            }
          }
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: data.response || 'I understand your request. Let me help you with that.',
          timestamp: new Date(),
          automation: {
            type: 'workflow_creation',
            status: 'running',
            progress: 0
          }
        };

        setMessages(prev => [...prev, aiMessage]);
        setActiveAutomation(aiMessage.id);

        // Simulate automation progress
        let progress = 0;
        const interval = setInterval(() => {
          progress += 10;
          setMessages(prev => prev.map(msg => 
            msg.id === aiMessage.id 
              ? { ...msg, automation: { ...msg.automation!, progress } }
              : msg
          ));

          if (progress >= 100) {
            clearInterval(interval);
            setActiveAutomation(null);
            setMessages(prev => prev.map(msg => 
              msg.id === aiMessage.id 
                ? { 
                    ...msg, 
                    automation: { ...msg.automation!, status: 'completed', progress: 100 },
                    sources: [
                      {
                        title: 'Automation Workflow Created',
                        url: 'https://example.com/workflow',
                        snippet: 'Your automation workflow has been successfully created and is ready for execution.',
                        domain: 'automation-platform.com',
                        relevance: 0.95,
                        source: 'system'
                      }
                    ],
                    files: [
                      {
                        name: 'workflow_report.pdf',
                        type: 'pdf',
                        size: '2.3 MB',
                        url: '/downloads/workflow_report.pdf'
                      }
                    ]
                  }
                : msg
            ));
          }
        }, 500);

      } else {
        throw new Error('Failed to get response from AI');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        status: 'error'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleAutomationControl = (action: string, messageId: string) => {
    console.log(`Automation control: ${action} for message ${messageId}`);
    
    if (action === 'play') {
      setActiveAutomation(messageId);
    } else if (action === 'pause') {
      setActiveAutomation(null);
    }
  };

  const handleUserInput = (messageId: string, data: any) => {
    console.log(`User input for message ${messageId}:`, data);
  };

  const handleCopyToClipboard = (text: string) => {
    console.log('Copied to clipboard:', text);
  };

  return (
    <div className="h-screen">
      <SimpleChatInterface
        messages={messages}
        isTyping={isTyping}
        activeAutomation={activeAutomation}
        onSendMessage={handleSendMessage}
        onAutomationControl={handleAutomationControl}
        onUserInput={handleUserInput}
        onCopyToClipboard={handleCopyToClipboard}
      />
    </div>
  );
}