/**
 * SUPER-OMEGA Edge Kernel - Background Service Worker
 * Provides sub-25ms automation execution with native driver integration
 */

import { EdgeKernelCore } from './edge-kernel-core.js';
import { MicroPlanner } from './micro-planner.js';
import { VisionProcessor } from './vision-processor.js';

class SuperOmegaBackground {
    constructor() {
        this.edgeKernel = new EdgeKernelCore();
        this.microPlanner = new MicroPlanner();
        this.visionProcessor = new VisionProcessor();
        this.activeSessions = new Map();
        this.performanceMonitor = new PerformanceMonitor();
        
        this.setupMessageHandlers();
        this.setupNativeMessaging();
        this.initializeKernel();
    }
    
    async initializeKernel() {
        try {
            // Initialize WASM modules for sub-25ms performance
            await this.edgeKernel.initialize();
            await this.microPlanner.loadModel();
            await this.visionProcessor.initialize();
            
            console.log('SUPER-OMEGA Edge Kernel initialized');
            
            // Start performance monitoring
            this.performanceMonitor.start();
            
        } catch (error) {
            console.error('Failed to initialize Edge Kernel:', error);
        }
    }
    
    setupMessageHandlers() {
        // Handle messages from content scripts and popup
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async response
        });
        
        // Handle external connections from native driver
        chrome.runtime.onConnectExternal.addListener((port) => {
            this.handleExternalConnection(port);
        });
        
        // Handle tab updates for DOM monitoring
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });
    }
    
    setupNativeMessaging() {
        // Connect to native Tauri driver
        try {
            this.nativePort = chrome.runtime.connectNative('com.super_omega.edge_kernel');
            
            this.nativePort.onMessage.addListener((message) => {
                this.handleNativeMessage(message);
            });
            
            this.nativePort.onDisconnect.addListener(() => {
                console.log('Native messaging disconnected');
                // Attempt reconnection
                setTimeout(() => this.setupNativeMessaging(), 1000);
            });
            
        } catch (error) {
            console.log('Native messaging not available, running in browser-only mode');
        }
    }
    
    async handleMessage(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            switch (message.type) {
                case 'CREATE_SESSION':
                    const sessionId = await this.createSession(message.config);
                    sendResponse({ success: true, sessionId });
                    break;
                    
                case 'EXECUTE_ACTION':
                    const result = await this.executeAction(
                        message.sessionId,
                        message.action,
                        sender.tab?.id
                    );
                    sendResponse({ success: true, result });
                    break;
                    
                case 'GET_DOM_SNAPSHOT':
                    const snapshot = await this.getDOMSnapshot(sender.tab?.id);
                    sendResponse({ success: true, snapshot });
                    break;
                    
                case 'ANALYZE_PAGE':
                    const analysis = await this.analyzePage(sender.tab?.id);
                    sendResponse({ success: true, analysis });
                    break;
                    
                case 'GET_PERFORMANCE_METRICS':
                    const metrics = this.performanceMonitor.getMetrics();
                    sendResponse({ success: true, metrics });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
            
        } catch (error) {
            console.error('Message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
        
        // Track performance
        const executionTime = performance.now() - startTime;
        this.performanceMonitor.recordExecution(message.type, executionTime);
        
        // Ensure sub-25ms response for critical operations
        if (['EXECUTE_ACTION', 'GET_DOM_SNAPSHOT'].includes(message.type) && executionTime > 25) {
            console.warn(`Operation ${message.type} took ${executionTime.toFixed(2)}ms, exceeding 25ms target`);
        }
    }
    
    async createSession(config) {
        const sessionId = this.generateSessionId();
        
        const session = {
            id: sessionId,
            config: config,
            startTime: Date.now(),
            actions: [],
            domGraph: new Map(),
            evidenceCollector: new EvidenceCollector(sessionId),
            performance: {
                totalActions: 0,
                successfulActions: 0,
                averageExecutionTime: 0,
                sub25msActions: 0
            }
        };
        
        this.activeSessions.set(sessionId, session);
        
        // Start evidence collection with proper /runs/<id>/ structure
        await session.evidenceCollector.startSession();
        
        return sessionId;
    }
    
    async executeAction(sessionId, action, tabId) {
        const session = this.activeSessions.get(sessionId);
        if (!session) {
            throw new Error('Session not found');
        }
        
        const startTime = performance.now();
        
        try {
            // Step 1: Micro-planner generates optimal strategy (target: sub-5ms)
            const strategy = await this.microPlanner.planAction(action, session.domGraph);
            
            // Step 2: Vision processor analyzes current state (target: sub-10ms)
            const visualContext = await this.visionProcessor.analyzeTab(tabId);
            
            // Step 3: Execute action with self-healing selectors (target: sub-10ms)
            const executionResult = await this.edgeKernel.executeAction(
                tabId,
                action,
                strategy,
                visualContext
            );
            
            // Step 4: Capture evidence (target: sub-8ms)
            await session.evidenceCollector.captureStepEvidence(
                session.actions.length + 1,
                action.type,
                action.selector,
                executionResult.success,
                performance.now() - startTime,
                executionResult.error
            );
            
            // Update session statistics
            session.actions.push({
                action,
                result: executionResult,
                executionTime: performance.now() - startTime,
                timestamp: Date.now()
            });
            
            this.updateSessionPerformance(session, performance.now() - startTime);
            
            return executionResult;
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            
            // Capture failed action evidence
            await session.evidenceCollector.captureStepEvidence(
                session.actions.length + 1,
                action.type,
                action.selector || 'unknown',
                false,
                executionTime,
                error.message
            );
            
            throw error;
        }
    }
    
    async getDOMSnapshot(tabId) {
        const startTime = performance.now();
        
        try {
            // Inject content script to capture comprehensive DOM
            const results = await chrome.scripting.executeScript({
                target: { tabId },
                func: this.captureComprehensiveDOM,
            });
            
            const domData = results[0]?.result;
            
            // Process with vision embeddings
            const processedDOM = await this.visionProcessor.processDOM(domData);
            
            const executionTime = performance.now() - startTime;
            
            // Ensure sub-5ms target for DOM capture
            if (executionTime > 5) {
                console.warn(`DOM snapshot took ${executionTime.toFixed(2)}ms, exceeding 5ms target`);
            }
            
            return processedDOM;
            
        } catch (error) {
            console.error('DOM snapshot failed:', error);
            throw error;
        }
    }
    
    // Injected function to capture comprehensive DOM
    captureComprehensiveDOM() {
        const startTime = performance.now();
        
        // Capture all interactive elements with detailed information
        const elements = Array.from(document.querySelectorAll('*'))
            .filter(el => {
                const style = getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden';
            })
            .slice(0, 5000) // Limit to prevent performance issues
            .map(el => {
                const rect = el.getBoundingClientRect();
                
                return {
                    tagName: el.tagName,
                    id: el.id,
                    className: el.className,
                    textContent: el.textContent?.substring(0, 200),
                    attributes: Array.from(el.attributes).reduce((acc, attr) => {
                        acc[attr.name] = attr.value;
                        return acc;
                    }, {}),
                    boundingRect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    },
                    xpath: this.generateXPath(el),
                    cssSelector: this.generateCSSSelector(el),
                    isInteractable: this.isInteractable(el),
                    role: el.getAttribute('role') || this.inferRole(el),
                    ariaLabel: el.getAttribute('aria-label'),
                    computedStyle: {
                        display: getComputedStyle(el).display,
                        position: getComputedStyle(el).position,
                        zIndex: getComputedStyle(el).zIndex
                    }
                };
            });
        
        return {
            timestamp: Date.now(),
            url: window.location.href,
            title: document.title,
            elements: elements,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            captureTime: performance.now() - startTime
        };
    }
    
    generateXPath(element) {
        if (element.id) {
            return `//*[@id="${element.id}"]`;
        }
        
        let path = '';
        let current = element;
        
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let index = 1;
            let sibling = current.previousElementSibling;
            
            while (sibling) {
                if (sibling.tagName === current.tagName) {
                    index++;
                }
                sibling = sibling.previousElementSibling;
            }
            
            const tagName = current.tagName.toLowerCase();
            path = `/${tagName}[${index}]${path}`;
            current = current.parentElement;
        }
        
        return path;
    }
    
    generateCSSSelector(element) {
        if (element.id) {
            return `#${element.id}`;
        }
        
        let selector = element.tagName.toLowerCase();
        
        if (element.className) {
            const classes = element.className.split(' ').filter(c => c);
            if (classes.length > 0) {
                selector += '.' + classes.join('.');
            }
        }
        
        // Make selector unique by adding parent context if needed
        let current = element.parentElement;
        while (current && document.querySelectorAll(selector).length > 1) {
            const parentSelector = current.tagName.toLowerCase();
            selector = `${parentSelector} > ${selector}`;
            current = current.parentElement;
        }
        
        return selector;
    }
    
    isInteractable(element) {
        const interactableTags = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'];
        const hasClickHandler = element.onclick || element.getAttribute('onclick');
        const hasRole = ['button', 'link', 'tab', 'menuitem'].includes(element.getAttribute('role'));
        
        return interactableTags.includes(element.tagName) || hasClickHandler || hasRole;
    }
    
    inferRole(element) {
        const tagRoles = {
            'A': 'link',
            'BUTTON': 'button',
            'INPUT': 'textbox',
            'SELECT': 'combobox',
            'TEXTAREA': 'textbox',
            'IMG': 'img'
        };
        
        return tagRoles[element.tagName] || 'generic';
    }
    
    async analyzePage(tabId) {
        // Comprehensive page analysis for automation opportunities
        const domSnapshot = await this.getDOMSnapshot(tabId);
        const visualAnalysis = await this.visionProcessor.analyzeTab(tabId);
        
        return {
            automation_opportunities: this.identifyAutomationOpportunities(domSnapshot),
            visual_elements: visualAnalysis.elements,
            page_structure: this.analyzePageStructure(domSnapshot),
            performance_score: this.calculatePageScore(domSnapshot)
        };
    }
    
    identifyAutomationOpportunities(domSnapshot) {
        const opportunities = [];
        
        // Identify forms
        const forms = domSnapshot.elements.filter(el => el.tagName === 'FORM');
        forms.forEach(form => {
            opportunities.push({
                type: 'form_fill',
                element: form,
                confidence: 0.95,
                actions: ['fill_form', 'submit']
            });
        });
        
        // Identify navigation elements
        const navElements = domSnapshot.elements.filter(el => 
            el.role === 'link' || el.tagName === 'A'
        );
        navElements.forEach(nav => {
            opportunities.push({
                type: 'navigation',
                element: nav,
                confidence: 0.90,
                actions: ['click', 'navigate']
            });
        });
        
        return opportunities;
    }
    
    updateSessionPerformance(session, executionTime) {
        session.performance.totalActions++;
        
        if (executionTime <= 25) {
            session.performance.sub25msActions++;
        }
        
        // Update rolling average
        const total = session.performance.totalActions;
        const current = session.performance.averageExecutionTime;
        session.performance.averageExecutionTime = 
            (current * (total - 1) + executionTime) / total;
    }
    
    handleNativeMessage(message) {
        // Handle messages from native Tauri driver
        switch (message.type) {
            case 'PERFORMANCE_UPDATE':
                this.performanceMonitor.updateFromNative(message.data);
                break;
                
            case 'MODEL_UPDATE':
                this.microPlanner.updateModel(message.modelData);
                break;
                
            default:
                console.log('Unknown native message:', message);
        }
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

class PerformanceMonitor {
    constructor() {
        this.metrics = {
            totalExecutions: 0,
            sub25msExecutions: 0,
            averageExecutionTime: 0,
            maxExecutionTime: 0,
            minExecutionTime: Infinity,
            operationTimes: new Map()
        };
    }
    
    start() {
        // Start continuous performance monitoring
        setInterval(() => {
            this.checkMemoryUsage();
        }, 5000);
    }
    
    recordExecution(operation, executionTime) {
        this.metrics.totalExecutions++;
        
        if (executionTime <= 25) {
            this.metrics.sub25msExecutions++;
        }
        
        // Update statistics
        this.metrics.maxExecutionTime = Math.max(this.metrics.maxExecutionTime, executionTime);
        this.metrics.minExecutionTime = Math.min(this.metrics.minExecutionTime, executionTime);
        
        // Update rolling average
        const total = this.metrics.totalExecutions;
        const current = this.metrics.averageExecutionTime;
        this.metrics.averageExecutionTime = (current * (total - 1) + executionTime) / total;
        
        // Track per-operation metrics
        if (!this.metrics.operationTimes.has(operation)) {
            this.metrics.operationTimes.set(operation, []);
        }
        
        const operationTimes = this.metrics.operationTimes.get(operation);
        operationTimes.push(executionTime);
        
        // Keep only last 100 measurements per operation
        if (operationTimes.length > 100) {
            operationTimes.shift();
        }
    }
    
    getMetrics() {
        return {
            ...this.metrics,
            sub25msRate: this.metrics.sub25msExecutions / Math.max(this.metrics.totalExecutions, 1),
            operationStats: this.getOperationStats()
        };
    }
    
    getOperationStats() {
        const stats = {};
        
        for (const [operation, times] of this.metrics.operationTimes) {
            const sorted = [...times].sort((a, b) => a - b);
            const avg = times.reduce((a, b) => a + b, 0) / times.length;
            
            stats[operation] = {
                count: times.length,
                average: avg,
                min: sorted[0],
                max: sorted[sorted.length - 1],
                p95: sorted[Math.floor(sorted.length * 0.95)],
                sub25msRate: times.filter(t => t <= 25).length / times.length
            };
        }
        
        return stats;
    }
    
    checkMemoryUsage() {
        if (performance.memory) {
            const memoryInfo = {
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            };
            
            // Warn if memory usage is high
            const usagePercent = memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit;
            if (usagePercent > 0.8) {
                console.warn(`High memory usage: ${(usagePercent * 100).toFixed(1)}%`);
            }
        }
    }
}

class EvidenceCollector {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.stepCounter = 0;
        this.evidenceData = [];
    }
    
    async startSession() {
        // Initialize evidence collection following /runs/<id>/ structure
        this.startTime = Date.now();
        
        // Store initial session data
        await chrome.storage.local.set({
            [`session_${this.sessionId}`]: {
                startTime: this.startTime,
                steps: [],
                performance: {}
            }
        });
    }
    
    async captureStepEvidence(stepNumber, actionType, selector, success, executionTime, error) {
        this.stepCounter++;
        
        const evidence = {
            step_number: stepNumber,
            timestamp: new Date().toISOString(),
            action_type: actionType,
            target_selector: selector,
            success: success,
            execution_time_ms: executionTime,
            error_message: error || null,
            performance_metrics: {
                response_time_ms: executionTime,
                memory_usage_mb: this.getMemoryUsage()
            }
        };
        
        this.evidenceData.push(evidence);
        
        // Store in chrome storage (simulating /runs/<id>/steps/<n>.json)
        await chrome.storage.local.set({
            [`session_${this.sessionId}_step_${stepNumber}`]: evidence
        });
        
        return evidence;
    }
    
    getMemoryUsage() {
        return performance.memory ? 
            performance.memory.usedJSHeapSize / 1024 / 1024 : 0;
    }
}

// Initialize the background service
const superOmegaBackground = new SuperOmegaBackground();