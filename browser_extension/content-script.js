/**
 * SUPER-OMEGA Edge Automation - Content Script
 * ============================================
 * 
 * Real-time DOM monitoring and interaction for edge-first automation.
 * Provides sub-25ms element detection and healing capabilities.
 */

class SuperOmegaContentScript {
    constructor() {
        this.domObserver = null;
        this.elementCache = new Map();
        this.selectorHealing = new SelectorHealing();
        this.performanceTracker = new PerformanceTracker();
        this.isInitialized = false;
        
        this.initialize();
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('🚀 SUPER-OMEGA Content Script initializing...');
        
        // Setup DOM observation
        this.setupDOMObserver();
        
        // Setup message listeners
        this.setupMessageListeners();
        
        // Cache initial DOM state
        await this.cacheInitialDOM();
        
        // Inject edge kernel
        this.injectEdgeKernel();
        
        this.isInitialized = true;
        console.log('✅ SUPER-OMEGA Content Script ready');
    }
    
    setupDOMObserver() {
        this.domObserver = new MutationObserver((mutations) => {
            this.handleDOMChanges(mutations);
        });
        
        this.domObserver.observe(document, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeOldValue: true,
            characterData: true
        });
    }
    
    setupMessageListeners() {
        // Listen for messages from background script
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open
        });
        
        // Listen for messages from injected script
        window.addEventListener('message', (event) => {
            if (event.source !== window) return;
            if (event.data.source === 'super-omega-injected') {
                this.handleInjectedMessage(event.data);
            }
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            switch (message.type) {
                case 'FIND_ELEMENT':
                    return await this.findElement(message, sendResponse);
                
                case 'CLICK_ELEMENT':
                    return await this.clickElement(message, sendResponse);
                
                case 'TYPE_TEXT':
                    return await this.typeText(message, sendResponse);
                
                case 'GET_DOM_STATE':
                    return await this.getDOMState(message, sendResponse);
                
                case 'HEAL_SELECTOR':
                    return await this.healSelector(message, sendResponse);
                
                case 'CAPTURE_SCREENSHOT':
                    return await this.captureScreenshot(message, sendResponse);
                
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution(message.type, executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                executionTime
            });
        }
    }
    
    async findElement(message, sendResponse) {
        const startTime = performance.now();
        const { selector, options = {} } = message.data;
        
        try {
            // Try direct selector first
            let element = document.querySelector(selector);
            let method = 'direct';
            
            if (!element && options.enableHealing) {
                // Use self-healing if element not found
                const healingResult = await this.selectorHealing.healSelector(selector);
                if (healingResult.success) {
                    element = document.querySelector(healingResult.selector);
                    method = 'healed';
                    selector = healingResult.selector;
                }
            }
            
            if (!element) {
                throw new Error(`Element not found: ${selector}`);
            }
            
            // Get element information
            const elementInfo = this.getElementInfo(element);
            const executionTime = performance.now() - startTime;
            
            this.performanceTracker.recordExecution('FIND_ELEMENT', executionTime, true);
            
            sendResponse({
                success: true,
                element: elementInfo,
                selector,
                method,
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('FIND_ELEMENT', executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                selector,
                executionTime
            });
        }
    }
    
    async clickElement(message, sendResponse) {
        const startTime = performance.now();
        const { selector, options = {} } = message.data;
        
        try {
            const element = await this.findElementWithHealing(selector, options);
            
            if (!element) {
                throw new Error(`Element not found for clicking: ${selector}`);
            }
            
            // Ensure element is visible and clickable
            this.ensureElementVisible(element);
            
            // Perform human-like click
            await this.performHumanLikeClick(element, options);
            
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('CLICK_ELEMENT', executionTime, true);
            
            sendResponse({
                success: true,
                action: 'click',
                selector,
                elementText: element.textContent?.slice(0, 100),
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('CLICK_ELEMENT', executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                selector,
                executionTime
            });
        }
    }
    
    async typeText(message, sendResponse) {
        const startTime = performance.now();
        const { selector, text, options = {} } = message.data;
        
        try {
            const element = await this.findElementWithHealing(selector, options);
            
            if (!element) {
                throw new Error(`Element not found for typing: ${selector}`);
            }
            
            // Ensure element is focusable
            this.ensureElementFocusable(element);
            
            // Perform human-like typing
            await this.performHumanLikeTyping(element, text, options);
            
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('TYPE_TEXT', executionTime, true);
            
            sendResponse({
                success: true,
                action: 'type',
                selector,
                text: text.slice(0, 100),
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('TYPE_TEXT', executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                selector,
                text,
                executionTime
            });
        }
    }
    
    async getDOMState(message, sendResponse) {
        const startTime = performance.now();
        
        try {
            const domState = {
                url: window.location.href,
                title: document.title,
                readyState: document.readyState,
                elementsCount: document.querySelectorAll('*').length,
                interactableElements: this.getInteractableElements(),
                formsCount: document.forms.length,
                linksCount: document.links.length,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    scrollX: window.scrollX,
                    scrollY: window.scrollY
                },
                timestamp: Date.now()
            };
            
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('GET_DOM_STATE', executionTime, true);
            
            sendResponse({
                success: true,
                domState,
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('GET_DOM_STATE', executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                executionTime
            });
        }
    }
    
    async healSelector(message, sendResponse) {
        const startTime = performance.now();
        const { originalSelector, context = {} } = message.data;
        
        try {
            const healingResult = await this.selectorHealing.healSelector(originalSelector, context);
            const executionTime = performance.now() - startTime;
            
            this.performanceTracker.recordExecution('HEAL_SELECTOR', executionTime, healingResult.success);
            
            sendResponse({
                success: healingResult.success,
                originalSelector,
                healedSelector: healingResult.selector,
                method: healingResult.method,
                confidence: healingResult.confidence,
                executionTime,
                sub25ms: executionTime < 25
            });
            
        } catch (error) {
            const executionTime = performance.now() - startTime;
            this.performanceTracker.recordExecution('HEAL_SELECTOR', executionTime, false);
            
            sendResponse({
                success: false,
                error: error.message,
                originalSelector,
                executionTime
            });
        }
    }
    
    async findElementWithHealing(selector, options = {}) {
        let element = document.querySelector(selector);
        
        if (!element && options.enableHealing !== false) {
            const healingResult = await this.selectorHealing.healSelector(selector);
            if (healingResult.success) {
                element = document.querySelector(healingResult.selector);
            }
        }
        
        return element;
    }
    
    getElementInfo(element) {
        const rect = element.getBoundingClientRect();
        
        return {
            tagName: element.tagName,
            id: element.id,
            className: element.className,
            textContent: element.textContent?.slice(0, 200),
            attributes: this.getElementAttributes(element),
            boundingRect: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            isVisible: this.isElementVisible(element),
            isInteractable: this.isElementInteractable(element),
            cssSelector: this.generateCSSSelector(element),
            xpath: this.generateXPath(element)
        };
    }
    
    getElementAttributes(element) {
        const attributes = {};
        for (const attr of element.attributes) {
            attributes[attr.name] = attr.value;
        }
        return attributes;
    }
    
    isElementVisible(element) {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        
        return (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            style.opacity !== '0' &&
            rect.width > 0 &&
            rect.height > 0 &&
            rect.top < window.innerHeight &&
            rect.bottom > 0 &&
            rect.left < window.innerWidth &&
            rect.right > 0
        );
    }
    
    isElementInteractable(element) {
        const interactableTags = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'];
        const hasClickHandler = element.onclick || element.getAttribute('onclick');
        const hasRole = ['button', 'link', 'tab', 'menuitem'].includes(element.getAttribute('role'));
        
        return (
            interactableTags.includes(element.tagName) ||
            hasClickHandler ||
            hasRole ||
            element.hasAttribute('tabindex')
        );
    }
    
    getInteractableElements() {
        const elements = Array.from(document.querySelectorAll('*'))
            .filter(el => this.isElementInteractable(el) && this.isElementVisible(el))
            .slice(0, 100) // Limit for performance
            .map(el => ({
                tagName: el.tagName,
                id: el.id,
                className: el.className,
                textContent: el.textContent?.slice(0, 50),
                selector: this.generateCSSSelector(el)
            }));
        
        return elements;
    }
    
    ensureElementVisible(element) {
        if (!this.isElementVisible(element)) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Wait for scroll to complete
            return new Promise(resolve => {
                setTimeout(resolve, 100);
            });
        }
    }
    
    ensureElementFocusable(element) {
        if (!element.hasAttribute('tabindex') && !['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName)) {
            element.setAttribute('tabindex', '-1');
        }
    }
    
    async performHumanLikeClick(element, options = {}) {
        // Focus the element first
        element.focus();
        
        // Add small delay to simulate human behavior
        await this.sleep(options.delay || 50);
        
        // Dispatch mouse events in sequence
        const rect = element.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        
        const mouseEvents = ['mousedown', 'mouseup', 'click'];
        
        for (const eventType of mouseEvents) {
            const event = new MouseEvent(eventType, {
                bubbles: true,
                cancelable: true,
                clientX: x,
                clientY: y,
                button: 0
            });
            
            element.dispatchEvent(event);
            
            if (eventType !== 'click') {
                await this.sleep(10);
            }
        }
    }
    
    async performHumanLikeTyping(element, text, options = {}) {
        // Focus the element
        element.focus();
        
        // Clear existing content if needed
        if (options.clearFirst !== false) {
            element.value = '';
        }
        
        // Type character by character with human-like delays
        const delay = options.typingDelay || 50;
        
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            
            // Dispatch keydown event
            const keydownEvent = new KeyboardEvent('keydown', {
                bubbles: true,
                cancelable: true,
                key: char,
                char: char,
                charCode: char.charCodeAt(0),
                keyCode: char.charCodeAt(0)
            });
            element.dispatchEvent(keydownEvent);
            
            // Update value
            element.value += char;
            
            // Dispatch input event
            const inputEvent = new Event('input', {
                bubbles: true,
                cancelable: true
            });
            element.dispatchEvent(inputEvent);
            
            // Dispatch keyup event
            const keyupEvent = new KeyboardEvent('keyup', {
                bubbles: true,
                cancelable: true,
                key: char,
                char: char,
                charCode: char.charCodeAt(0),
                keyCode: char.charCodeAt(0)
            });
            element.dispatchEvent(keyupEvent);
            
            // Human-like delay between characters
            if (i < text.length - 1) {
                await this.sleep(delay + Math.random() * 20);
            }
        }
        
        // Dispatch change event
        const changeEvent = new Event('change', {
            bubbles: true,
            cancelable: true
        });
        element.dispatchEvent(changeEvent);
    }
    
    generateCSSSelector(element) {
        if (element.id) {
            return `#${element.id}`;
        }
        
        let selector = element.tagName.toLowerCase();
        
        if (element.className) {
            const classes = element.className.split(' ').filter(c => c && !c.includes(' '));
            if (classes.length > 0) {
                selector += '.' + classes.slice(0, 2).join('.');
            }
        }
        
        // Make selector unique if needed
        if (document.querySelectorAll(selector).length > 1) {
            let parent = element.parentElement;
            let parentSelector = '';
            
            while (parent && document.querySelectorAll(parentSelector + selector).length > 1) {
                const parentTag = parent.tagName.toLowerCase();
                const parentClass = parent.className ? '.' + parent.className.split(' ')[0] : '';
                parentSelector = `${parentTag}${parentClass} > ` + parentSelector;
                parent = parent.parentElement;
            }
            
            selector = parentSelector + selector;
        }
        
        return selector;
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
    
    async cacheInitialDOM() {
        const interactableElements = this.getInteractableElements();
        
        for (const elementInfo of interactableElements) {
            this.elementCache.set(elementInfo.selector, {
                ...elementInfo,
                timestamp: Date.now()
            });
        }
    }
    
    handleDOMChanges(mutations) {
        // Update element cache based on DOM changes
        for (const mutation of mutations) {
            if (mutation.type === 'childList') {
                // Handle added/removed nodes
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.cacheNewElement(node);
                    }
                }
            }
        }
    }
    
    cacheNewElement(element) {
        if (this.isElementInteractable(element) && this.isElementVisible(element)) {
            const selector = this.generateCSSSelector(element);
            this.elementCache.set(selector, {
                tagName: element.tagName,
                id: element.id,
                className: element.className,
                textContent: element.textContent?.slice(0, 50),
                selector,
                timestamp: Date.now()
            });
        }
    }
    
    injectEdgeKernel() {
        // Inject edge kernel script for enhanced performance
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('edge-kernel.js');
        script.onload = () => {
            console.log('✅ Edge Kernel injected');
        };
        (document.head || document.documentElement).appendChild(script);
    }
    
    handleInjectedMessage(data) {
        // Handle messages from injected edge kernel
        switch (data.type) {
            case 'PERFORMANCE_UPDATE':
                this.performanceTracker.updateFromInjected(data.metrics);
                break;
                
            case 'DOM_ANALYSIS':
                this.updateDOMAnalysis(data.analysis);
                break;
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Selector Healing Implementation
class SelectorHealing {
    constructor() {
        this.healingStrategies = [
            this.healByTextContent.bind(this),
            this.healByAttributes.bind(this),
            this.healByPosition.bind(this),
            this.healByParentContext.bind(this),
            this.healByGenericSelector.bind(this)
        ];
    }
    
    async healSelector(originalSelector, context = {}) {
        console.log('🔧 Attempting to heal selector:', originalSelector);
        
        for (const strategy of this.healingStrategies) {
            try {
                const result = await strategy(originalSelector, context);
                if (result.success) {
                    console.log('✅ Selector healed using:', result.method);
                    return result;
                }
            } catch (error) {
                console.warn('Healing strategy failed:', error);
            }
        }
        
        return {
            success: false,
            error: 'All healing strategies failed',
            originalSelector
        };
    }
    
    async healByTextContent(originalSelector, context) {
        // Extract text content from original element if available
        const originalElement = document.querySelector(originalSelector);
        if (!originalElement) {
            return { success: false };
        }
        
        const textContent = originalElement.textContent?.trim();
        if (!textContent) {
            return { success: false };
        }
        
        // Find elements with similar text content
        const candidates = Array.from(document.querySelectorAll('*'))
            .filter(el => {
                const elText = el.textContent?.trim();
                return elText && elText.includes(textContent.slice(0, 20));
            });
        
        if (candidates.length === 1) {
            const healedSelector = this.generateSelectorForElement(candidates[0]);
            return {
                success: true,
                selector: healedSelector,
                method: 'text_content',
                confidence: 0.8
            };
        }
        
        return { success: false };
    }
    
    async healByAttributes(originalSelector, context) {
        // Extract attributes from original selector
        const attributes = this.extractAttributesFromSelector(originalSelector);
        
        if (Object.keys(attributes).length === 0) {
            return { success: false };
        }
        
        // Try different attribute combinations
        for (const [attr, value] of Object.entries(attributes)) {
            const selector = `[${attr}="${value}"]`;
            if (document.querySelector(selector)) {
                return {
                    success: true,
                    selector,
                    method: 'attributes',
                    confidence: 0.9
                };
            }
            
            // Try partial attribute matching
            const partialSelector = `[${attr}*="${value}"]`;
            if (document.querySelector(partialSelector)) {
                return {
                    success: true,
                    selector: partialSelector,
                    method: 'partial_attributes',
                    confidence: 0.7
                };
            }
        }
        
        return { success: false };
    }
    
    async healByPosition(originalSelector, context) {
        // Try to find elements in similar positions
        const tagName = this.extractTagFromSelector(originalSelector);
        
        if (!tagName) {
            return { success: false };
        }
        
        const elements = Array.from(document.querySelectorAll(tagName));
        
        if (elements.length === 1) {
            return {
                success: true,
                selector: tagName,
                method: 'tag_name',
                confidence: 0.6
            };
        }
        
        // Try with nth-child selectors
        for (let i = 1; i <= Math.min(elements.length, 5); i++) {
            const selector = `${tagName}:nth-child(${i})`;
            if (document.querySelector(selector)) {
                return {
                    success: true,
                    selector,
                    method: 'position',
                    confidence: 0.5
                };
            }
        }
        
        return { success: false };
    }
    
    async healByParentContext(originalSelector, context) {
        // Try to find elements within similar parent contexts
        const tagName = this.extractTagFromSelector(originalSelector);
        
        if (!tagName) {
            return { success: false };
        }
        
        const commonParents = ['form', 'div', 'section', 'main', 'article'];
        
        for (const parent of commonParents) {
            const selector = `${parent} ${tagName}`;
            const elements = document.querySelectorAll(selector);
            
            if (elements.length === 1) {
                return {
                    success: true,
                    selector,
                    method: 'parent_context',
                    confidence: 0.7
                };
            }
        }
        
        return { success: false };
    }
    
    async healByGenericSelector(originalSelector, context) {
        // Generic fallback selectors based on common patterns
        const genericSelectors = this.generateGenericSelectors(originalSelector);
        
        for (const selector of genericSelectors) {
            if (document.querySelector(selector)) {
                return {
                    success: true,
                    selector,
                    method: 'generic_fallback',
                    confidence: 0.4
                };
            }
        }
        
        return { success: false };
    }
    
    generateGenericSelectors(originalSelector) {
        const selectors = [];
        
        if (originalSelector.includes('button') || originalSelector.includes('btn')) {
            selectors.push('button', 'input[type="submit"]', '[role="button"]', '.btn', '.button');
        }
        
        if (originalSelector.includes('input') || originalSelector.includes('textbox')) {
            selectors.push('input', 'textarea', '[role="textbox"]', '[contenteditable="true"]');
        }
        
        if (originalSelector.includes('link') || originalSelector.includes('a')) {
            selectors.push('a', '[role="link"]');
        }
        
        if (originalSelector.includes('search')) {
            selectors.push('input[type="search"]', '[placeholder*="search"]', 'input[name*="search"]', '#search');
        }
        
        return selectors;
    }
    
    generateSelectorForElement(element) {
        if (element.id) {
            return `#${element.id}`;
        }
        
        let selector = element.tagName.toLowerCase();
        
        if (element.className) {
            const classes = element.className.split(' ').filter(c => c);
            if (classes.length > 0) {
                selector += '.' + classes[0];
            }
        }
        
        return selector;
    }
    
    extractAttributesFromSelector(selector) {
        const attributes = {};
        
        // Extract ID
        const idMatch = selector.match(/#([^.\[\s]+)/);
        if (idMatch) {
            attributes.id = idMatch[1];
        }
        
        // Extract classes
        const classMatches = selector.match(/\.([^.\[\s]+)/g);
        if (classMatches) {
            attributes.class = classMatches.map(c => c.substring(1)).join(' ');
        }
        
        // Extract attributes
        const attrMatches = selector.match(/\[([^=]+)=["']([^"']+)["']\]/g);
        if (attrMatches) {
            for (const match of attrMatches) {
                const [, name, value] = match.match(/\[([^=]+)=["']([^"']+)["']\]/);
                attributes[name] = value;
            }
        }
        
        return attributes;
    }
    
    extractTagFromSelector(selector) {
        const match = selector.match(/^([a-zA-Z]+)/);
        return match ? match[1] : null;
    }
}

// Performance Tracker Implementation
class PerformanceTracker {
    constructor() {
        this.metrics = {
            totalOperations: 0,
            successfulOperations: 0,
            sub25msOperations: 0,
            averageExecutionTime: 0,
            operationHistory: []
        };
    }
    
    recordExecution(operation, executionTime, success) {
        this.metrics.totalOperations++;
        
        if (success) {
            this.metrics.successfulOperations++;
        }
        
        if (executionTime < 25) {
            this.metrics.sub25msOperations++;
        }
        
        // Update rolling average
        const total = this.metrics.totalOperations;
        const current = this.metrics.averageExecutionTime;
        this.metrics.averageExecutionTime = (current * (total - 1) + executionTime) / total;
        
        // Store operation history (keep last 100)
        this.metrics.operationHistory.push({
            operation,
            executionTime,
            success,
            timestamp: Date.now()
        });
        
        if (this.metrics.operationHistory.length > 100) {
            this.metrics.operationHistory.shift();
        }
        
        // Report to background if needed
        if (this.metrics.totalOperations % 10 === 0) {
            this.reportToBackground();
        }
    }
    
    getMetrics() {
        return {
            ...this.metrics,
            successRate: this.metrics.successfulOperations / Math.max(this.metrics.totalOperations, 1),
            sub25msRate: this.metrics.sub25msOperations / Math.max(this.metrics.totalOperations, 1)
        };
    }
    
    reportToBackground() {
        chrome.runtime.sendMessage({
            type: 'PERFORMANCE_UPDATE',
            data: this.getMetrics()
        }).catch(() => {
            // Ignore errors if background script is not available
        });
    }
    
    updateFromInjected(metrics) {
        // Update metrics from injected script
        this.metrics = { ...this.metrics, ...metrics };
    }
}

// Initialize content script
const superOmegaContentScript = new SuperOmegaContentScript();