/**
 * SUPER-OMEGA Edge Automation - Background Service Worker
 * ======================================================
 * 
 * Edge-first execution with sub-25ms decisions and native driver integration.
 * Handles communication between extension, content scripts, and native host.
 */

import { EdgeKernel } from './edge-kernel.js';
import { NativeHostManager } from './native-host-manager.js';

class SuperOmegaBackground {
    constructor() {
        this.edgeKernel = new EdgeKernel();
        this.nativeHost = new NativeHostManager();
        this.activeSessions = new Map();
        this.performanceMetrics = {
            decisionsTotal: 0,
            decisionsUnder25ms: 0,
            averageDecisionTime: 0,
            successRate: 0
        };
        
        this.initializeExtension();
    }
    
    async initializeExtension() {
        console.log('🚀 SUPER-OMEGA Background Service Worker initializing...');
        
        // Initialize edge kernel
        await this.edgeKernel.initialize();
        
        // Setup native host connection
        await this.nativeHost.initialize();
        
        // Setup event listeners
        this.setupEventListeners();
        
        console.log('✅ SUPER-OMEGA Background Service Worker ready');
    }
    
    setupEventListeners() {
        // Extension messages
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async response
        });
        
        // Tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });
        
        // Native messaging
        chrome.runtime.onConnectExternal.addListener((port) => {
            this.handleExternalConnection(port);
        });
        
        // Extension installation/startup
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            console.log('📨 Background received message:', message.type);
            
            switch (message.type) {
                case 'EXECUTE_AUTOMATION':
                    return await this.executeAutomation(message, sender, sendResponse);
                
                case 'START_SESSION':
                    return await this.startAutomationSession(message, sender, sendResponse);
                
                case 'STOP_SESSION':
                    return await this.stopAutomationSession(message, sender, sendResponse);
                
                case 'HEAL_SELECTOR':
                    return await this.healSelector(message, sender, sendResponse);
                
                case 'GET_PERFORMANCE_STATS':
                    return await this.getPerformanceStats(message, sender, sendResponse);
                
                case 'PING':
                    return sendResponse({ success: true, timestamp: Date.now() });
                
                default:
                    console.warn('Unknown message type:', message.type);
                    return sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Message handling error:', error);
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetrics(executionTime, false);
            
            return sendResponse({ 
                success: false, 
                error: error.message,
                executionTime
            });
        }
    }
    
    async executeAutomation(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            const { instruction, options = {} } = message.data;
            const tabId = sender.tab?.id;
            
            if (!tabId) {
                throw new Error('No tab ID available');
            }
            
            console.log('🎯 Executing automation:', instruction);
            
            // Use edge kernel for sub-25ms decision making
            const decision = await this.edgeKernel.makeDecision({
                instruction,
                tabId,
                options,
                context: await this.getTabContext(tabId)
            });
            
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetrics(executionTime, decision.success);
            
            // Execute the decision
            const result = await this.executeDecision(decision, tabId);
            
            sendResponse({
                success: true,
                result,
                decision,
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetrics(executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                executionTime
            });
        }
    }
    
    async startAutomationSession(message, sender, sendResponse) {
        try {
            const { sessionId, config = {} } = message.data;
            const tabId = sender.tab?.id;
            
            if (!tabId) {
                throw new Error('No tab ID available');
            }
            
            console.log('🎬 Starting automation session:', sessionId);
            
            const session = {
                id: sessionId,
                tabId,
                config,
                startTime: Date.now(),
                actions: [],
                status: 'active',
                performance: {
                    decisionsTotal: 0,
                    decisionsUnder25ms: 0,
                    successRate: 0
                }
            };
            
            this.activeSessions.set(sessionId, session);
            
            // Inject content script if needed
            await this.ensureContentScript(tabId);
            
            // Notify native host
            await this.nativeHost.sendMessage({
                type: 'SESSION_STARTED',
                sessionId,
                tabId,
                config
            });
            
            sendResponse({
                success: true,
                sessionId,
                status: 'active'
            });
            
        } catch (error) {
            sendResponse({
                success: false,
                error: error.message
            });
        }
    }
    
    async stopAutomationSession(message, sender, sendResponse) {
        try {
            const { sessionId } = message.data;
            const session = this.activeSessions.get(sessionId);
            
            if (!session) {
                throw new Error(`Session ${sessionId} not found`);
            }
            
            console.log('🛑 Stopping automation session:', sessionId);
            
            session.status = 'stopped';
            session.endTime = Date.now();
            
            // Notify native host
            await this.nativeHost.sendMessage({
                type: 'SESSION_STOPPED',
                sessionId,
                duration: session.endTime - session.startTime,
                performance: session.performance
            });
            
            this.activeSessions.delete(sessionId);
            
            sendResponse({
                success: true,
                sessionId,
                status: 'stopped',
                duration: session.endTime - session.startTime
            });
            
        } catch (error) {
            sendResponse({
                success: false,
                error: error.message
            });
        }
    }
    
    async healSelector(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            const { originalSelector, context = {} } = message.data;
            const tabId = sender.tab?.id;
            
            if (!tabId) {
                throw new Error('No tab ID available');
            }
            
            console.log('🔧 Healing selector:', originalSelector);
            
            // Use edge kernel for fast selector healing
            const healingResult = await this.edgeKernel.healSelector({
                originalSelector,
                tabId,
                context
            });
            
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetrics(executionTime, healingResult.success);
            
            sendResponse({
                success: healingResult.success,
                healedSelector: healingResult.selector,
                method: healingResult.method,
                confidence: healingResult.confidence,
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetrics(executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                originalSelector: message.data.originalSelector,
                executionTime
            });
        }
    }
    
    async getPerformanceStats(message, sender, sendResponse) {
        try {
            const stats = {
                ...this.performanceMetrics,
                activeSessions: this.activeSessions.size,
                edgeKernelStats: await this.edgeKernel.getStats(),
                nativeHostConnected: this.nativeHost.isConnected(),
                timestamp: Date.now()
            };
            
            sendResponse({
                success: true,
                stats
            });
            
        } catch (error) {
            sendResponse({
                success: false,
                error: error.message
            });
        }
    }
    
    async executeDecision(decision, tabId) {
        try {
            switch (decision.action) {
                case 'click':
                    return await this.executeClick(decision, tabId);
                
                case 'type':
                    return await this.executeType(decision, tabId);
                
                case 'navigate':
                    return await this.executeNavigate(decision, tabId);
                
                case 'wait':
                    return await this.executeWait(decision, tabId);
                
                case 'heal_and_retry':
                    return await this.executeHealAndRetry(decision, tabId);
                
                default:
                    throw new Error(`Unknown action: ${decision.action}`);
            }
        } catch (error) {
            console.error('Decision execution failed:', error);
            throw error;
        }
    }
    
    async executeClick(decision, tabId) {
        return await chrome.scripting.executeScript({
            target: { tabId },
            func: (selector) => {
                const element = document.querySelector(selector);
                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }
                
                element.click();
                return {
                    success: true,
                    action: 'click',
                    selector,
                    elementText: element.textContent?.slice(0, 100)
                };
            },
            args: [decision.selector]
        });
    }
    
    async executeType(decision, tabId) {
        return await chrome.scripting.executeScript({
            target: { tabId },
            func: (selector, text) => {
                const element = document.querySelector(selector);
                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }
                
                element.focus();
                element.value = text;
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
                
                return {
                    success: true,
                    action: 'type',
                    selector,
                    text: text.slice(0, 100)
                };
            },
            args: [decision.selector, decision.text]
        });
    }
    
    async executeNavigate(decision, tabId) {
        await chrome.tabs.update(tabId, { url: decision.url });
        
        return {
            success: true,
            action: 'navigate',
            url: decision.url
        };
    }
    
    async executeWait(decision, tabId) {
        await new Promise(resolve => setTimeout(resolve, decision.duration || 1000));
        
        return {
            success: true,
            action: 'wait',
            duration: decision.duration || 1000
        };
    }
    
    async executeHealAndRetry(decision, tabId) {
        // Attempt to heal the selector first
        const healingResult = await this.edgeKernel.healSelector({
            originalSelector: decision.originalSelector,
            tabId,
            context: decision.context
        });
        
        if (healingResult.success) {
            // Retry the original action with healed selector
            const newDecision = {
                ...decision,
                selector: healingResult.selector,
                action: decision.originalAction
            };
            
            return await this.executeDecision(newDecision, tabId);
        } else {
            throw new Error('Selector healing failed');
        }
    }
    
    async getTabContext(tabId) {
        try {
            const [result] = await chrome.scripting.executeScript({
                target: { tabId },
                func: () => {
                    return {
                        url: window.location.href,
                        title: document.title,
                        readyState: document.readyState,
                        activeElement: document.activeElement?.tagName,
                        elementsCount: document.querySelectorAll('*').length,
                        formsCount: document.forms.length,
                        linksCount: document.links.length
                    };
                }
            });
            
            return result.result;
        } catch (error) {
            console.warn('Failed to get tab context:', error);
            return {};
        }
    }
    
    async ensureContentScript(tabId) {
        try {
            await chrome.scripting.executeScript({
                target: { tabId },
                files: ['content-script.js']
            });
        } catch (error) {
            console.warn('Content script injection failed:', error);
        }
    }
    
    updatePerformanceMetrics(executionTime, success) {
        this.performanceMetrics.decisionsTotal++;
        
        if (executionTime < 25) {
            this.performanceMetrics.decisionsUnder25ms++;
        }
        
        // Update average decision time
        const total = this.performanceMetrics.decisionsTotal;
        const currentAvg = this.performanceMetrics.averageDecisionTime;
        this.performanceMetrics.averageDecisionTime = 
            (currentAvg * (total - 1) + executionTime) / total;
        
        // Update success rate
        if (success) {
            const currentSuccesses = this.performanceMetrics.successRate * (total - 1);
            this.performanceMetrics.successRate = (currentSuccesses + 1) / total;
        } else {
            const currentSuccesses = this.performanceMetrics.successRate * (total - 1);
            this.performanceMetrics.successRate = currentSuccesses / total;
        }
    }
    
    handleTabUpdate(tabId, changeInfo, tab) {
        if (changeInfo.status === 'complete') {
            // Notify active sessions about page load completion
            for (const [sessionId, session] of this.activeSessions) {
                if (session.tabId === tabId) {
                    this.nativeHost.sendMessage({
                        type: 'PAGE_LOADED',
                        sessionId,
                        tabId,
                        url: tab.url
                    });
                }
            }
        }
    }
    
    handleExternalConnection(port) {
        console.log('🔌 External connection established:', port.name);
        
        port.onMessage.addListener((message) => {
            console.log('📥 External message:', message);
            // Handle external messages from native host or other applications
        });
        
        port.onDisconnect.addListener(() => {
            console.log('🔌 External connection disconnected');
        });
    }
    
    handleInstallation(details) {
        console.log('📦 Extension installed/updated:', details.reason);
        
        if (details.reason === 'install') {
            // First installation
            chrome.tabs.create({
                url: 'chrome-extension://' + chrome.runtime.id + '/welcome.html'
            });
        } else if (details.reason === 'update') {
            // Extension updated
            console.log('🔄 Extension updated to version:', chrome.runtime.getManifest().version);
        }
    }
}

// Edge Kernel Implementation
class EdgeKernel {
    constructor() {
        this.decisionCache = new Map();
        this.selectorDatabase = new Map();
        this.performanceHistory = [];
    }
    
    async initialize() {
        console.log('⚡ Edge Kernel initializing...');
        
        // Load selector database
        await this.loadSelectorDatabase();
        
        // Initialize decision trees
        this.initializeDecisionTrees();
        
        console.log('✅ Edge Kernel ready');
    }
    
    async makeDecision(context) {
        const startTime = performance.now();
        
        try {
            // Check cache first
            const cacheKey = this.generateCacheKey(context);
            if (this.decisionCache.has(cacheKey)) {
                const cachedDecision = this.decisionCache.get(cacheKey);
                const executionTime = performance.now() - startTime;
                
                return {
                    ...cachedDecision,
                    cached: true,
                    executionTime,
                    sub25ms: executionTime < 25
                };
            }
            
            // Make new decision
            const decision = await this.analyzeAndDecide(context);
            
            // Cache the decision
            this.decisionCache.set(cacheKey, decision);
            
            const executionTime = performance.now() - startTime;
            
            return {
                ...decision,
                cached: false,
                executionTime,
                sub25ms: executionTime < 25
            };
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            
            return {
                success: false,
                error: error.message,
                action: 'fallback',
                executionTime,
                sub25ms: executionTime < 25
            };
        }
    }
    
    async analyzeAndDecide(context) {
        const { instruction, tabId, options, context: tabContext } = context;
        
        // Simple instruction parsing for demo
        const instructionLower = instruction.toLowerCase();
        
        if (instructionLower.includes('click')) {
            return {
                success: true,
                action: 'click',
                selector: this.extractSelector(instruction) || 'button',
                confidence: 0.8
            };
        } else if (instructionLower.includes('type') || instructionLower.includes('enter')) {
            return {
                success: true,
                action: 'type',
                selector: 'input',
                text: this.extractText(instruction) || 'test',
                confidence: 0.8
            };
        } else if (instructionLower.includes('navigate') || instructionLower.includes('go to')) {
            return {
                success: true,
                action: 'navigate',
                url: this.extractUrl(instruction) || 'https://www.google.com',
                confidence: 0.8
            };
        } else {
            return {
                success: true,
                action: 'wait',
                duration: 1000,
                confidence: 0.5
            };
        }
    }
    
    async healSelector(context) {
        const { originalSelector, tabId } = context;
        
        try {
            // Try to find alternative selectors
            const alternatives = this.selectorDatabase.get(originalSelector) || [];
            
            for (const alternative of alternatives) {
                // Test if alternative selector works
                const [result] = await chrome.scripting.executeScript({
                    target: { tabId },
                    func: (selector) => {
                        return document.querySelector(selector) !== null;
                    },
                    args: [alternative]
                });
                
                if (result.result) {
                    return {
                        success: true,
                        selector: alternative,
                        method: 'database_lookup',
                        confidence: 0.9
                    };
                }
            }
            
            // Fallback to generic selectors
            const genericSelectors = this.generateGenericSelectors(originalSelector);
            
            for (const genericSelector of genericSelectors) {
                const [result] = await chrome.scripting.executeScript({
                    target: { tabId },
                    func: (selector) => {
                        return document.querySelector(selector) !== null;
                    },
                    args: [genericSelector]
                });
                
                if (result.result) {
                    return {
                        success: true,
                        selector: genericSelector,
                        method: 'generic_fallback',
                        confidence: 0.6
                    };
                }
            }
            
            return {
                success: false,
                error: 'No working selector found',
                originalSelector
            };
            
        } catch (error) {
            return {
                success: false,
                error: error.message,
                originalSelector
            };
        }
    }
    
    async loadSelectorDatabase() {
        // Load common selector alternatives
        const commonSelectors = {
            'button': ['input[type="submit"]', '[role="button"]', '.btn', '.button'],
            'input': ['[role="textbox"]', 'textarea', '[contenteditable="true"]'],
            'a': ['[role="link"]', 'button[onclick*="location"]'],
            '.search': ['[placeholder*="search"]', 'input[name*="search"]', '#search']
        };
        
        for (const [key, alternatives] of Object.entries(commonSelectors)) {
            this.selectorDatabase.set(key, alternatives);
        }
    }
    
    initializeDecisionTrees() {
        // Initialize decision trees for fast decision making
        this.decisionTrees = {
            'click': {
                confidence: 0.9,
                fallbacks: ['wait', 'heal_and_retry']
            },
            'type': {
                confidence: 0.8,
                fallbacks: ['click', 'heal_and_retry']
            },
            'navigate': {
                confidence: 0.95,
                fallbacks: ['wait']
            }
        };
    }
    
    generateCacheKey(context) {
        const { instruction, options } = context;
        return `${instruction}_${JSON.stringify(options)}`;
    }
    
    extractSelector(instruction) {
        // Simple selector extraction
        const selectorMatch = instruction.match(/['"`]([^'"`]+)['"`]/);
        return selectorMatch ? selectorMatch[1] : null;
    }
    
    extractText(instruction) {
        // Simple text extraction
        const textMatch = instruction.match(/type\s+['"`]([^'"`]+)['"`]/i);
        return textMatch ? textMatch[1] : null;
    }
    
    extractUrl(instruction) {
        // Simple URL extraction
        const urlMatch = instruction.match(/https?:\/\/[^\s]+/);
        return urlMatch ? urlMatch[0] : null;
    }
    
    generateGenericSelectors(originalSelector) {
        // Generate generic fallback selectors
        const selectors = [];
        
        if (originalSelector.includes('button')) {
            selectors.push('button', 'input[type="submit"]', '[role="button"]');
        }
        
        if (originalSelector.includes('input')) {
            selectors.push('input', 'textarea', '[role="textbox"]');
        }
        
        if (originalSelector.includes('link') || originalSelector.includes('a')) {
            selectors.push('a', '[role="link"]');
        }
        
        return selectors;
    }
    
    async getStats() {
        return {
            cacheSize: this.decisionCache.size,
            selectorDatabaseSize: this.selectorDatabase.size,
            performanceHistorySize: this.performanceHistory.length
        };
    }
}

// Native Host Manager Implementation
class NativeHostManager {
    constructor() {
        this.port = null;
        this.connected = false;
        this.messageQueue = [];
    }
    
    async initialize() {
        try {
            console.log('🔌 Connecting to native host...');
            
            this.port = chrome.runtime.connectNative('com.superomega.nativehost');
            
            this.port.onMessage.addListener((message) => {
                this.handleNativeMessage(message);
            });
            
            this.port.onDisconnect.addListener(() => {
                console.log('🔌 Native host disconnected');
                this.connected = false;
                this.port = null;
            });
            
            // Send initial handshake
            await this.sendMessage({
                type: 'HANDSHAKE',
                version: chrome.runtime.getManifest().version,
                timestamp: Date.now()
            });
            
            this.connected = true;
            console.log('✅ Native host connected');
            
        } catch (error) {
            console.warn('⚠️ Native host connection failed:', error);
            this.connected = false;
        }
    }
    
    async sendMessage(message) {
        if (this.connected && this.port) {
            try {
                this.port.postMessage(message);
                return true;
            } catch (error) {
                console.error('Native host message send failed:', error);
                return false;
            }
        } else {
            // Queue message for later
            this.messageQueue.push(message);
            return false;
        }
    }
    
    handleNativeMessage(message) {
        console.log('📥 Native host message:', message);
        
        switch (message.type) {
            case 'HANDSHAKE_ACK':
                console.log('🤝 Native host handshake acknowledged');
                break;
                
            case 'PERFORMANCE_UPDATE':
                console.log('📊 Native host performance update:', message.data);
                break;
                
            default:
                console.log('Unknown native host message:', message);
        }
    }
    
    isConnected() {
        return this.connected;
    }
}

// Initialize the background service worker
const superOmegaBackground = new SuperOmegaBackground();