/**
 * SUPER-OMEGA Edge Kernel - Content Script
 * Provides real-time DOM monitoring and sub-25ms action execution
 */

class SuperOmegaContentScript {
    constructor() {
        this.domObserver = null;
        this.performanceMonitor = new ContentPerformanceMonitor();
        this.actionExecutor = new ActionExecutor();
        this.domGraph = new Map();
        this.sessionId = null;
        
        this.setupMessageHandlers();
        this.initializeDOMMonitoring();
        this.injectPageScript();
    }
    
    setupMessageHandlers() {
        // Listen for messages from background script
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open
        });
        
        // Listen for messages from injected script
        window.addEventListener('message', (event) => {
            if (event.source !== window || !event.data.type?.startsWith('SUPER_OMEGA_')) {
                return;
            }
            
            this.handlePageMessage(event.data);
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            switch (message.type) {
                case 'EXECUTE_ACTION':
                    const result = await this.executeAction(message.action);
                    sendResponse({ success: true, result });
                    break;
                    
                case 'GET_DOM_ELEMENTS':
                    const elements = await this.getDOMElements(message.selector);
                    sendResponse({ success: true, elements });
                    break;
                    
                case 'CAPTURE_SCREENSHOT':
                    const screenshot = await this.captureElementScreenshot(message.selector);
                    sendResponse({ success: true, screenshot });
                    break;
                    
                case 'ANALYZE_ELEMENT':
                    const analysis = await this.analyzeElement(message.selector);
                    sendResponse({ success: true, analysis });
                    break;
                    
                case 'START_MONITORING':
                    this.sessionId = message.sessionId;
                    this.startRealTimeMonitoring();
                    sendResponse({ success: true });
                    break;
                    
                case 'STOP_MONITORING':
                    this.stopRealTimeMonitoring();
                    sendResponse({ success: true });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
            
        } catch (error) {
            console.error('Content script error:', error);
            sendResponse({ success: false, error: error.message });
        }
        
        // Track performance - ensure sub-25ms for critical operations
        const executionTime = performance.now() - startTime;
        if (['EXECUTE_ACTION', 'GET_DOM_ELEMENTS'].includes(message.type) && executionTime > 25) {
            console.warn(`Content script operation ${message.type} took ${executionTime.toFixed(2)}ms`);
        }
    }
    
    async executeAction(action) {
        const startTime = performance.now();
        
        try {
            // Find target element with self-healing selectors
            const element = await this.findElementWithFallbacks(action);
            
            if (!element) {
                throw new Error(`Element not found: ${action.selector}`);
            }
            
            // Execute action based on type
            let result;
            switch (action.type) {
                case 'click':
                    result = await this.actionExecutor.click(element, action);
                    break;
                    
                case 'type':
                    result = await this.actionExecutor.type(element, action.text, action);
                    break;
                    
                case 'select':
                    result = await this.actionExecutor.select(element, action.value, action);
                    break;
                    
                case 'hover':
                    result = await this.actionExecutor.hover(element, action);
                    break;
                    
                case 'scroll':
                    result = await this.actionExecutor.scroll(element, action);
                    break;
                    
                default:
                    throw new Error(`Unknown action type: ${action.type}`);
            }
            
            // Record successful execution
            const executionTime = performance.now() - startTime;
            this.performanceMonitor.recordAction(action.type, executionTime, true);
            
            return {
                success: true,
                executionTime: executionTime,
                elementInfo: this.getElementInfo(element),
                result: result
            };
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceMonitor.recordAction(action.type, executionTime, false);
            
            // Attempt self-healing
            if (action.enableSelfHealing !== false) {
                const healedResult = await this.attemptSelfHealing(action, error);
                if (healedResult) {
                    return healedResult;
                }
            }
            
            throw error;
        }
    }
    
    async findElementWithFallbacks(action) {
        const selectors = [
            action.selector,
            ...(action.fallbackSelectors || [])
        ];
        
        for (const selector of selectors) {
            try {
                // Try different selector strategies
                let element = null;
                
                // CSS selector
                if (selector.startsWith('#') || selector.startsWith('.') || selector.includes('[')) {
                    element = document.querySelector(selector);
                }
                
                // XPath selector
                else if (selector.startsWith('/') || selector.startsWith('(')) {
                    const result = document.evaluate(
                        selector,
                        document,
                        null,
                        XPathResult.FIRST_ORDERED_NODE_TYPE,
                        null
                    );
                    element = result.singleNodeValue;
                }
                
                // Text-based selector
                else if (selector.startsWith('text:')) {
                    const text = selector.substring(5);
                    element = this.findElementByText(text);
                }
                
                // Role-based selector
                else if (selector.startsWith('role:')) {
                    const role = selector.substring(5);
                    element = document.querySelector(`[role="${role}"]`);
                }
                
                if (element && this.isElementVisible(element)) {
                    return element;
                }
                
            } catch (error) {
                console.debug(`Selector failed: ${selector}`, error);
                continue;
            }
        }
        
        return null;
    }
    
    findElementByText(text) {
        const xpath = `//*[contains(text(), "${text}")]`;
        const result = document.evaluate(
            xpath,
            document,
            null,
            XPathResult.FIRST_ORDERED_NODE_TYPE,
            null
        );
        return result.singleNodeValue;
    }
    
    isElementVisible(element) {
        if (!element) return false;
        
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        
        return (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            style.opacity !== '0' &&
            rect.width > 0 &&
            rect.height > 0 &&
            rect.top >= 0 &&
            rect.left >= 0
        );
    }
    
    async attemptSelfHealing(action, originalError) {
        console.log('Attempting self-healing for failed action:', action);
        
        // Strategy 1: Find similar elements by text content
        if (action.expectedText) {
            const similarElements = this.findElementsBySimilarText(action.expectedText);
            for (const element of similarElements) {
                try {
                    const result = await this.executeActionOnElement(element, action);
                    console.log('Self-healing successful using similar text');
                    return result;
                } catch (error) {
                    continue;
                }
            }
        }
        
        // Strategy 2: Find elements by role and position
        if (action.role && action.position) {
            const roleElements = document.querySelectorAll(`[role="${action.role}"]`);
            const targetElement = Array.from(roleElements)[action.position];
            
            if (targetElement) {
                try {
                    const result = await this.executeActionOnElement(targetElement, action);
                    console.log('Self-healing successful using role and position');
                    return result;
                } catch (error) {
                    // Continue to next strategy
                }
            }
        }
        
        // Strategy 3: Visual similarity (if vision embeddings available)
        if (action.visualHash) {
            const visuallyLikelyElements = await this.findVisuallyLikelyElements(action.visualHash);
            for (const element of visuallyLikelyElements) {
                try {
                    const result = await this.executeActionOnElement(element, action);
                    console.log('Self-healing successful using visual similarity');
                    return result;
                } catch (error) {
                    continue;
                }
            }
        }
        
        return null; // Self-healing failed
    }
    
    findElementsBySimilarText(expectedText) {
        const allElements = document.querySelectorAll('*');
        const similar = [];
        
        for (const element of allElements) {
            const text = element.textContent?.trim();
            if (text && this.calculateTextSimilarity(text, expectedText) > 0.8) {
                similar.push(element);
            }
        }
        
        return similar.sort((a, b) => {
            const scoreA = this.calculateTextSimilarity(a.textContent, expectedText);
            const scoreB = this.calculateTextSimilarity(b.textContent, expectedText);
            return scoreB - scoreA;
        });
    }
    
    calculateTextSimilarity(text1, text2) {
        // Simple Levenshtein distance-based similarity
        const longer = text1.length > text2.length ? text1 : text2;
        const shorter = text1.length > text2.length ? text2 : text1;
        
        if (longer.length === 0) return 1.0;
        
        const distance = this.levenshteinDistance(longer, shorter);
        return (longer.length - distance) / longer.length;
    }
    
    levenshteinDistance(str1, str2) {
        const matrix = [];
        
        for (let i = 0; i <= str2.length; i++) {
            matrix[i] = [i];
        }
        
        for (let j = 0; j <= str1.length; j++) {
            matrix[0][j] = j;
        }
        
        for (let i = 1; i <= str2.length; i++) {
            for (let j = 1; j <= str1.length; j++) {
                if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }
        
        return matrix[str2.length][str1.length];
    }
    
    async executeActionOnElement(element, action) {
        switch (action.type) {
            case 'click':
                return await this.actionExecutor.click(element, action);
            case 'type':
                return await this.actionExecutor.type(element, action.text, action);
            case 'select':
                return await this.actionExecutor.select(element, action.value, action);
            default:
                throw new Error(`Unsupported action type: ${action.type}`);
        }
    }
    
    getElementInfo(element) {
        const rect = element.getBoundingClientRect();
        
        return {
            tagName: element.tagName,
            id: element.id,
            className: element.className,
            textContent: element.textContent?.substring(0, 100),
            attributes: Array.from(element.attributes).reduce((acc, attr) => {
                acc[attr.name] = attr.value;
                return acc;
            }, {}),
            boundingRect: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            xpath: this.generateXPath(element),
            cssSelector: this.generateCSSSelector(element)
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
        
        return selector;
    }
    
    initializeDOMMonitoring() {
        // Set up mutation observer for real-time DOM changes
        this.domObserver = new MutationObserver((mutations) => {
            this.handleDOMChanges(mutations);
        });
        
        this.domObserver.observe(document, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeOldValue: true,
            characterData: true,
            characterDataOldValue: true
        });
    }
    
    handleDOMChanges(mutations) {
        if (!this.sessionId) return;
        
        const changes = [];
        
        for (const mutation of mutations) {
            changes.push({
                type: mutation.type,
                target: {
                    tagName: mutation.target.tagName,
                    id: mutation.target.id,
                    className: mutation.target.className
                },
                addedNodes: Array.from(mutation.addedNodes).map(node => ({
                    nodeType: node.nodeType,
                    tagName: node.tagName,
                    textContent: node.textContent?.substring(0, 50)
                })),
                removedNodes: Array.from(mutation.removedNodes).map(node => ({
                    nodeType: node.nodeType,
                    tagName: node.tagName,
                    textContent: node.textContent?.substring(0, 50)
                })),
                attributeName: mutation.attributeName,
                oldValue: mutation.oldValue
            });
        }
        
        // Send DOM changes to background script
        chrome.runtime.sendMessage({
            type: 'DOM_CHANGES',
            sessionId: this.sessionId,
            changes: changes,
            timestamp: Date.now()
        });
    }
    
    startRealTimeMonitoring() {
        // Start performance monitoring
        this.performanceMonitor.start();
        
        // Monitor page performance
        this.monitorPagePerformance();
        
        // Monitor network activity
        this.monitorNetworkActivity();
    }
    
    stopRealTimeMonitoring() {
        this.performanceMonitor.stop();
    }
    
    monitorPagePerformance() {
        // Monitor Core Web Vitals and other performance metrics
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                chrome.runtime.sendMessage({
                    type: 'PERFORMANCE_METRIC',
                    sessionId: this.sessionId,
                    metric: {
                        name: entry.name,
                        entryType: entry.entryType,
                        startTime: entry.startTime,
                        duration: entry.duration,
                        value: entry.value
                    },
                    timestamp: Date.now()
                });
            }
        });
        
        // Observe various performance metrics
        try {
            observer.observe({ entryTypes: ['navigation', 'resource', 'measure', 'paint'] });
        } catch (error) {
            console.debug('Some performance metrics not supported:', error);
        }
    }
    
    monitorNetworkActivity() {
        // Monitor fetch requests
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const startTime = performance.now();
            
            try {
                const response = await originalFetch(...args);
                const endTime = performance.now();
                
                chrome.runtime.sendMessage({
                    type: 'NETWORK_REQUEST',
                    sessionId: this.sessionId,
                    request: {
                        url: args[0],
                        method: args[1]?.method || 'GET',
                        status: response.status,
                        duration: endTime - startTime
                    },
                    timestamp: Date.now()
                });
                
                return response;
            } catch (error) {
                const endTime = performance.now();
                
                chrome.runtime.sendMessage({
                    type: 'NETWORK_REQUEST',
                    sessionId: this.sessionId,
                    request: {
                        url: args[0],
                        method: args[1]?.method || 'GET',
                        status: 'error',
                        error: error.message,
                        duration: endTime - startTime
                    },
                    timestamp: Date.now()
                });
                
                throw error;
            }
        };
    }
    
    injectPageScript() {
        // Inject script for deeper page access
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('injected.js');
        script.onload = function() {
            this.remove();
        };
        (document.head || document.documentElement).appendChild(script);
    }
    
    handlePageMessage(data) {
        // Handle messages from injected script
        switch (data.type) {
            case 'SUPER_OMEGA_PAGE_READY':
                chrome.runtime.sendMessage({
                    type: 'PAGE_READY',
                    sessionId: this.sessionId,
                    pageInfo: data.pageInfo
                });
                break;
                
            case 'SUPER_OMEGA_USER_INTERACTION':
                chrome.runtime.sendMessage({
                    type: 'USER_INTERACTION',
                    sessionId: this.sessionId,
                    interaction: data.interaction
                });
                break;
        }
    }
}

class ActionExecutor {
    async click(element, action) {
        const startTime = performance.now();
        
        // Ensure element is in viewport
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Wait for element to be ready
        await this.waitForElementReady(element);
        
        // Perform click with proper event simulation
        const clickEvent = new MouseEvent('click', {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: element.getBoundingClientRect().left + element.getBoundingClientRect().width / 2,
            clientY: element.getBoundingClientRect().top + element.getBoundingClientRect().height / 2
        });
        
        element.dispatchEvent(clickEvent);
        
        // Also trigger native click for compatibility
        element.click();
        
        const executionTime = performance.now() - startTime;
        
        return {
            success: true,
            executionTime: executionTime,
            action: 'click'
        };
    }
    
    async type(element, text, action) {
        const startTime = performance.now();
        
        // Focus element
        element.focus();
        
        // Clear existing content if specified
        if (action.clearFirst !== false) {
            element.select();
            element.value = '';
        }
        
        // Type text with realistic timing
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            
            // Simulate keydown, keypress, keyup events
            const keydownEvent = new KeyboardEvent('keydown', {
                key: char,
                bubbles: true,
                cancelable: true
            });
            
            const keypressEvent = new KeyboardEvent('keypress', {
                key: char,
                bubbles: true,
                cancelable: true
            });
            
            const keyupEvent = new KeyboardEvent('keyup', {
                key: char,
                bubbles: true,
                cancelable: true
            });
            
            element.dispatchEvent(keydownEvent);
            element.dispatchEvent(keypressEvent);
            
            // Update value
            element.value += char;
            
            // Trigger input event
            const inputEvent = new Event('input', {
                bubbles: true,
                cancelable: true
            });
            element.dispatchEvent(inputEvent);
            
            element.dispatchEvent(keyupEvent);
            
            // Add small delay for realism (if not in fast mode)
            if (!action.fastTyping) {
                await new Promise(resolve => setTimeout(resolve, Math.random() * 50 + 10));
            }
        }
        
        // Trigger change event
        const changeEvent = new Event('change', {
            bubbles: true,
            cancelable: true
        });
        element.dispatchEvent(changeEvent);
        
        const executionTime = performance.now() - startTime;
        
        return {
            success: true,
            executionTime: executionTime,
            action: 'type',
            text: text
        };
    }
    
    async select(element, value, action) {
        const startTime = performance.now();
        
        element.focus();
        
        // For select elements
        if (element.tagName === 'SELECT') {
            element.value = value;
            
            const changeEvent = new Event('change', {
                bubbles: true,
                cancelable: true
            });
            element.dispatchEvent(changeEvent);
        }
        // For other elements, try to find and click option
        else {
            const option = element.querySelector(`option[value="${value}"]`) ||
                          element.querySelector(`option:contains("${value}")`);
            
            if (option) {
                option.selected = true;
                element.dispatchEvent(new Event('change', { bubbles: true }));
            } else {
                throw new Error(`Option not found: ${value}`);
            }
        }
        
        const executionTime = performance.now() - startTime;
        
        return {
            success: true,
            executionTime: executionTime,
            action: 'select',
            value: value
        };
    }
    
    async hover(element, action) {
        const startTime = performance.now();
        
        const rect = element.getBoundingClientRect();
        const hoverEvent = new MouseEvent('mouseover', {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: rect.left + rect.width / 2,
            clientY: rect.top + rect.height / 2
        });
        
        element.dispatchEvent(hoverEvent);
        
        const executionTime = performance.now() - startTime;
        
        return {
            success: true,
            executionTime: executionTime,
            action: 'hover'
        };
    }
    
    async scroll(element, action) {
        const startTime = performance.now();
        
        const scrollOptions = {
            behavior: action.smooth ? 'smooth' : 'auto',
            block: action.block || 'center',
            inline: action.inline || 'center'
        };
        
        element.scrollIntoView(scrollOptions);
        
        const executionTime = performance.now() - startTime;
        
        return {
            success: true,
            executionTime: executionTime,
            action: 'scroll'
        };
    }
    
    async waitForElementReady(element, timeout = 5000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            if (this.isElementReady(element)) {
                return true;
            }
            
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        
        throw new Error('Element not ready within timeout');
    }
    
    isElementReady(element) {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        
        return (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            style.pointerEvents !== 'none' &&
            rect.width > 0 &&
            rect.height > 0 &&
            !element.disabled
        );
    }
}

class ContentPerformanceMonitor {
    constructor() {
        this.metrics = {
            actions: [],
            domChanges: 0,
            networkRequests: 0,
            startTime: Date.now()
        };
        
        this.isMonitoring = false;
    }
    
    start() {
        this.isMonitoring = true;
        this.startTime = Date.now();
        
        // Monitor frame rate
        this.monitorFrameRate();
    }
    
    stop() {
        this.isMonitoring = false;
    }
    
    recordAction(actionType, executionTime, success) {
        this.metrics.actions.push({
            type: actionType,
            executionTime: executionTime,
            success: success,
            timestamp: Date.now()
        });
        
        // Keep only last 100 actions
        if (this.metrics.actions.length > 100) {
            this.metrics.actions.shift();
        }
    }
    
    monitorFrameRate() {
        if (!this.isMonitoring) return;
        
        let frames = 0;
        let lastTime = performance.now();
        
        const countFrame = () => {
            frames++;
            const currentTime = performance.now();
            
            if (currentTime >= lastTime + 1000) {
                const fps = Math.round((frames * 1000) / (currentTime - lastTime));
                
                chrome.runtime.sendMessage({
                    type: 'FRAME_RATE',
                    fps: fps,
                    timestamp: Date.now()
                });
                
                frames = 0;
                lastTime = currentTime;
            }
            
            if (this.isMonitoring) {
                requestAnimationFrame(countFrame);
            }
        };
        
        requestAnimationFrame(countFrame);
    }
    
    getMetrics() {
        const now = Date.now();
        const totalTime = now - this.startTime;
        
        const successfulActions = this.metrics.actions.filter(a => a.success).length;
        const totalActions = this.metrics.actions.length;
        
        return {
            uptime: totalTime,
            totalActions: totalActions,
            successfulActions: successfulActions,
            successRate: totalActions > 0 ? successfulActions / totalActions : 0,
            averageActionTime: totalActions > 0 ? 
                this.metrics.actions.reduce((sum, a) => sum + a.executionTime, 0) / totalActions : 0,
            domChanges: this.metrics.domChanges,
            networkRequests: this.metrics.networkRequests
        };
    }
}

// Initialize content script
const superOmegaContent = new SuperOmegaContentScript();

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SuperOmegaContentScript, ActionExecutor, ContentPerformanceMonitor };
}