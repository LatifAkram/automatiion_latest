/**
 * SUPER-OMEGA Edge Kernel - Injected Script
 * =========================================
 * 
 * Ultra-fast DOM operations and decision making for sub-25ms performance.
 * This script runs in the page context for maximum performance.
 */

(function() {
    'use strict';
    
    // Prevent multiple injections
    if (window.SuperOmegaEdgeKernel) {
        return;
    }
    
    class SuperOmegaEdgeKernel {
        constructor() {
            this.elementCache = new Map();
            this.selectorCache = new Map();
            this.performanceMetrics = {
                operationsTotal: 0,
                operationsUnder25ms: 0,
                averageExecutionTime: 0,
                cacheHitRate: 0
            };
            
            this.initialize();
        }
        
        initialize() {
            console.log('⚡ SUPER-OMEGA Edge Kernel initializing...');
            
            // Setup ultra-fast element caching
            this.setupElementCaching();
            
            // Setup performance monitoring
            this.setupPerformanceMonitoring();
            
            // Setup message handling
            this.setupMessageHandling();
            
            // Pre-cache critical elements
            this.preCacheCriticalElements();
            
            console.log('✅ SUPER-OMEGA Edge Kernel ready');
            
            // Notify content script
            this.postMessage({
                type: 'KERNEL_READY',
                timestamp: performance.now()
            });
        }
        
        setupElementCaching() {
            // Ultra-fast element caching with MutationObserver
            this.cacheObserver = new MutationObserver((mutations) => {
                this.updateElementCache(mutations);
            });
            
            this.cacheObserver.observe(document, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['id', 'class', 'name', 'data-testid']
            });
        }
        
        setupPerformanceMonitoring() {
            // Ultra-lightweight performance monitoring
            this.performanceInterval = setInterval(() => {
                this.reportPerformanceMetrics();
            }, 5000);
        }
        
        setupMessageHandling() {
            // Listen for messages from content script
            window.addEventListener('message', (event) => {
                if (event.source !== window) return;
                if (event.data.target === 'super-omega-edge-kernel') {
                    this.handleMessage(event.data);
                }
            });
        }
        
        handleMessage(message) {
            const startTime = performance.now();
            
            try {
                switch (message.type) {
                    case 'FIND_ELEMENT_FAST':
                        return this.findElementFast(message.selector, startTime);
                    
                    case 'CLICK_ELEMENT_FAST':
                        return this.clickElementFast(message.selector, startTime);
                    
                    case 'TYPE_TEXT_FAST':
                        return this.typeTextFast(message.selector, message.text, startTime);
                    
                    case 'GET_CACHED_ELEMENTS':
                        return this.getCachedElements(startTime);
                    
                    case 'HEAL_SELECTOR_FAST':
                        return this.healSelectorFast(message.selector, startTime);
                    
                    default:
                        console.warn('Unknown edge kernel message:', message.type);
                }
            } catch (error) {
                const executionTime = performance.now() - startTime;
                this.recordPerformance(message.type, executionTime, false);
                
                this.postMessage({
                    type: 'ERROR',
                    error: error.message,
                    executionTime
                });
            }
        }
        
        findElementFast(selector, startTime) {
            try {
                // Check cache first (target: sub-1ms)
                const cacheKey = `find_${selector}`;
                if (this.selectorCache.has(cacheKey)) {
                    const cachedResult = this.selectorCache.get(cacheKey);
                    const element = document.querySelector(cachedResult.selector);
                    
                    if (element) {
                        const executionTime = performance.now() - startTime;
                        this.recordPerformance('FIND_ELEMENT_FAST', executionTime, true, true);
                        
                        this.postMessage({
                            type: 'ELEMENT_FOUND',
                            element: this.getElementInfo(element),
                            selector: cachedResult.selector,
                            method: 'cache_hit',
                            executionTime,
                            sub25ms: executionTime < 25
                        });
                        return;
                    }
                }
                
                // Direct query (target: sub-5ms)
                let element = document.querySelector(selector);
                let method = 'direct';
                let finalSelector = selector;
                
                // Fast healing if element not found (target: sub-15ms)
                if (!element) {
                    const healingResult = this.fastSelectorHealing(selector);
                    if (healingResult.success) {
                        element = document.querySelector(healingResult.selector);
                        method = 'healed';
                        finalSelector = healingResult.selector;
                    }
                }
                
                const executionTime = performance.now() - startTime;
                const success = element !== null;
                
                // Cache successful result
                if (success && method === 'healed') {
                    this.selectorCache.set(cacheKey, {
                        selector: finalSelector,
                        timestamp: Date.now()
                    });
                }
                
                this.recordPerformance('FIND_ELEMENT_FAST', executionTime, success, false);
                
                if (success) {
                    this.postMessage({
                        type: 'ELEMENT_FOUND',
                        element: this.getElementInfo(element),
                        selector: finalSelector,
                        method,
                        executionTime,
                        sub25ms: executionTime < 25
                    });
                } else {
                    this.postMessage({
                        type: 'ELEMENT_NOT_FOUND',
                        selector,
                        executionTime,
                        sub25ms: executionTime < 25
                    });
                }
                
            } catch (error) {
                const executionTime = performance.now() - startTime;
                this.recordPerformance('FIND_ELEMENT_FAST', executionTime, false, false);
                throw error;
            }
        }
        
        clickElementFast(selector, startTime) {
            try {
                // Find element with caching
                const element = this.findElementSync(selector);
                
                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }
                
                // Ultra-fast click (target: sub-10ms)
                this.performFastClick(element);
                
                const executionTime = performance.now() - startTime;
                this.recordPerformance('CLICK_ELEMENT_FAST', executionTime, true, false);
                
                this.postMessage({
                    type: 'ELEMENT_CLICKED',
                    selector,
                    elementText: element.textContent?.slice(0, 50),
                    executionTime,
                    sub25ms: executionTime < 25
                });
                
            } catch (error) {
                const executionTime = performance.now() - startTime;
                this.recordPerformance('CLICK_ELEMENT_FAST', executionTime, false, false);
                throw error;
            }
        }
        
        typeTextFast(selector, text, startTime) {
            try {
                // Find element with caching
                const element = this.findElementSync(selector);
                
                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }
                
                // Ultra-fast typing (target: sub-15ms)
                this.performFastTyping(element, text);
                
                const executionTime = performance.now() - startTime;
                this.recordPerformance('TYPE_TEXT_FAST', executionTime, true, false);
                
                this.postMessage({
                    type: 'TEXT_TYPED',
                    selector,
                    text: text.slice(0, 50),
                    executionTime,
                    sub25ms: executionTime < 25
                });
                
            } catch (error) {
                const executionTime = performance.now() - startTime;
                this.recordPerformance('TYPE_TEXT_FAST', executionTime, false, false);
                throw error;
            }
        }
        
        findElementSync(selector) {
            // Synchronous element finding with caching
            const cacheKey = `sync_${selector}`;
            
            if (this.elementCache.has(cacheKey)) {
                const cached = this.elementCache.get(cacheKey);
                if (Date.now() - cached.timestamp < 1000) { // 1 second cache
                    return cached.element;
                }
            }
            
            let element = document.querySelector(selector);
            
            // Fast healing if needed
            if (!element) {
                const healingResult = this.fastSelectorHealing(selector);
                if (healingResult.success) {
                    element = document.querySelector(healingResult.selector);
                }
            }
            
            // Cache the result
            if (element) {
                this.elementCache.set(cacheKey, {
                    element,
                    timestamp: Date.now()
                });
            }
            
            return element;
        }
        
        performFastClick(element) {
            // Ultra-fast click implementation
            element.focus();
            
            // Direct property manipulation for speed
            const event = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
                button: 0,
                buttons: 1,
                clientX: element.getBoundingClientRect().left + element.getBoundingClientRect().width / 2,
                clientY: element.getBoundingClientRect().top + element.getBoundingClientRect().height / 2
            });
            
            element.dispatchEvent(event);
        }
        
        performFastTyping(element, text) {
            // Ultra-fast typing implementation
            element.focus();
            
            // Direct value assignment for maximum speed
            element.value = text;
            
            // Dispatch only essential events
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }
        
        fastSelectorHealing(originalSelector) {
            // Ultra-fast selector healing (target: sub-10ms)
            const healingStrategies = [
                () => this.healByIdFallback(originalSelector),
                () => this.healByClassFallback(originalSelector),
                () => this.healByTagName(originalSelector),
                () => this.healByRole(originalSelector),
                () => this.healByGeneric(originalSelector)
            ];
            
            for (const strategy of healingStrategies) {
                try {
                    const result = strategy();
                    if (result.success) {
                        return result;
                    }
                } catch (error) {
                    // Continue to next strategy
                }
            }
            
            return { success: false };
        }
        
        healByIdFallback(selector) {
            // Extract ID and try variations
            const idMatch = selector.match(/#([^.\[\s]+)/);
            if (idMatch) {
                const id = idMatch[1];
                const variations = [
                    `#${id}`,
                    `[id="${id}"]`,
                    `[id*="${id}"]`,
                    `*[id="${id}"]`
                ];
                
                for (const variation of variations) {
                    if (document.querySelector(variation)) {
                        return {
                            success: true,
                            selector: variation,
                            method: 'id_fallback'
                        };
                    }
                }
            }
            
            return { success: false };
        }
        
        healByClassFallback(selector) {
            // Extract classes and try variations
            const classMatches = selector.match(/\.([^.\[\s]+)/g);
            if (classMatches) {
                const classes = classMatches.map(c => c.substring(1));
                
                for (const cls of classes) {
                    const variations = [
                        `.${cls}`,
                        `[class*="${cls}"]`,
                        `*[class*="${cls}"]`
                    ];
                    
                    for (const variation of variations) {
                        if (document.querySelector(variation)) {
                            return {
                                success: true,
                                selector: variation,
                                method: 'class_fallback'
                            };
                        }
                    }
                }
            }
            
            return { success: false };
        }
        
        healByTagName(selector) {
            // Extract tag name and try simple selectors
            const tagMatch = selector.match(/^([a-zA-Z]+)/);
            if (tagMatch) {
                const tag = tagMatch[1].toLowerCase();
                const elements = document.querySelectorAll(tag);
                
                if (elements.length === 1) {
                    return {
                        success: true,
                        selector: tag,
                        method: 'tag_name'
                    };
                }
                
                // Try with common parent contexts
                const contexts = ['form', 'div', 'main'];
                for (const context of contexts) {
                    const contextSelector = `${context} ${tag}`;
                    if (document.querySelector(contextSelector)) {
                        return {
                            success: true,
                            selector: contextSelector,
                            method: 'tag_with_context'
                        };
                    }
                }
            }
            
            return { success: false };
        }
        
        healByRole(selector) {
            // Try role-based selectors
            if (selector.includes('button')) {
                const roleSelectors = ['[role="button"]', 'button', 'input[type="submit"]'];
                for (const roleSelector of roleSelectors) {
                    if (document.querySelector(roleSelector)) {
                        return {
                            success: true,
                            selector: roleSelector,
                            method: 'role_based'
                        };
                    }
                }
            }
            
            if (selector.includes('input') || selector.includes('textbox')) {
                const inputSelectors = ['input', '[role="textbox"]', 'textarea'];
                for (const inputSelector of inputSelectors) {
                    if (document.querySelector(inputSelector)) {
                        return {
                            success: true,
                            selector: inputSelector,
                            method: 'role_based'
                        };
                    }
                }
            }
            
            return { success: false };
        }
        
        healByGeneric(selector) {
            // Generic fallback selectors
            const genericSelectors = [
                '*[id]',
                '*[class]',
                '*[name]',
                '*[data-testid]',
                'button',
                'input',
                'a',
                'div'
            ];
            
            for (const genericSelector of genericSelectors) {
                const elements = document.querySelectorAll(genericSelector);
                if (elements.length === 1) {
                    return {
                        success: true,
                        selector: genericSelector,
                        method: 'generic_fallback'
                    };
                }
            }
            
            return { success: false };
        }
        
        preCacheCriticalElements() {
            // Pre-cache commonly used elements for ultra-fast access
            const criticalSelectors = [
                'input[type="text"]',
                'input[type="email"]',
                'input[type="password"]',
                'input[type="search"]',
                'button[type="submit"]',
                'button',
                'a[href]',
                'form',
                '[role="button"]',
                '[role="textbox"]',
                '[role="link"]'
            ];
            
            for (const selector of criticalSelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    this.elementCache.set(`precache_${selector}`, {
                        elements: Array.from(elements),
                        timestamp: Date.now()
                    });
                }
            }
        }
        
        updateElementCache(mutations) {
            // Ultra-fast cache updates based on DOM mutations
            for (const mutation of mutations) {
                if (mutation.type === 'childList') {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.cacheNewElement(node);
                        }
                    }
                }
            }
        }
        
        cacheNewElement(element) {
            // Cache new interactive elements
            if (this.isInteractiveElement(element)) {
                const selector = this.generateFastSelector(element);
                this.elementCache.set(`new_${selector}`, {
                    element,
                    timestamp: Date.now()
                });
            }
        }
        
        isInteractiveElement(element) {
            const interactiveTags = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'];
            const hasRole = element.hasAttribute('role');
            const hasClickHandler = element.onclick || element.hasAttribute('onclick');
            
            return (
                interactiveTags.includes(element.tagName) ||
                hasRole ||
                hasClickHandler
            );
        }
        
        generateFastSelector(element) {
            // Generate selector optimized for speed
            if (element.id) {
                return `#${element.id}`;
            }
            
            if (element.name) {
                return `[name="${element.name}"]`;
            }
            
            if (element.className) {
                const firstClass = element.className.split(' ')[0];
                if (firstClass) {
                    return `.${firstClass}`;
                }
            }
            
            return element.tagName.toLowerCase();
        }
        
        getElementInfo(element) {
            // Ultra-fast element info extraction
            const rect = element.getBoundingClientRect();
            
            return {
                tagName: element.tagName,
                id: element.id,
                className: element.className,
                textContent: element.textContent?.slice(0, 100),
                boundingRect: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                },
                isVisible: rect.width > 0 && rect.height > 0,
                timestamp: Date.now()
            };
        }
        
        recordPerformance(operation, executionTime, success, fromCache) {
            this.performanceMetrics.operationsTotal++;
            
            if (executionTime < 25) {
                this.performanceMetrics.operationsUnder25ms++;
            }
            
            if (fromCache) {
                // Update cache hit rate
                const total = this.performanceMetrics.operationsTotal;
                const currentHits = this.performanceMetrics.cacheHitRate * (total - 1);
                this.performanceMetrics.cacheHitRate = (currentHits + 1) / total;
            }
            
            // Update rolling average
            const total = this.performanceMetrics.operationsTotal;
            const current = this.performanceMetrics.averageExecutionTime;
            this.performanceMetrics.averageExecutionTime = 
                (current * (total - 1) + executionTime) / total;
        }
        
        reportPerformanceMetrics() {
            this.postMessage({
                type: 'PERFORMANCE_UPDATE',
                metrics: {
                    ...this.performanceMetrics,
                    sub25msRate: this.performanceMetrics.operationsUnder25ms / Math.max(this.performanceMetrics.operationsTotal, 1),
                    cacheSize: this.elementCache.size,
                    selectorCacheSize: this.selectorCache.size
                }
            });
        }
        
        getCachedElements(startTime) {
            const cachedElements = [];
            
            for (const [key, value] of this.elementCache.entries()) {
                if (key.startsWith('precache_') || key.startsWith('new_')) {
                    if (value.elements) {
                        // Multiple elements
                        cachedElements.push({
                            selector: key.replace(/^(precache_|new_)/, ''),
                            count: value.elements.length,
                            timestamp: value.timestamp
                        });
                    } else if (value.element) {
                        // Single element
                        cachedElements.push({
                            selector: key.replace(/^(precache_|new_)/, ''),
                            count: 1,
                            timestamp: value.timestamp
                        });
                    }
                }
            }
            
            const executionTime = performance.now() - startTime;
            this.recordPerformance('GET_CACHED_ELEMENTS', executionTime, true, true);
            
            this.postMessage({
                type: 'CACHED_ELEMENTS',
                elements: cachedElements,
                executionTime,
                sub25ms: executionTime < 25
            });
        }
        
        healSelectorFast(selector, startTime) {
            const healingResult = this.fastSelectorHealing(selector);
            const executionTime = performance.now() - startTime;
            
            this.recordPerformance('HEAL_SELECTOR_FAST', executionTime, healingResult.success, false);
            
            this.postMessage({
                type: 'SELECTOR_HEALED',
                originalSelector: selector,
                healedSelector: healingResult.selector,
                method: healingResult.method,
                success: healingResult.success,
                executionTime,
                sub25ms: executionTime < 25
            });
        }
        
        postMessage(data) {
            // Post message to content script
            window.postMessage({
                source: 'super-omega-injected',
                ...data
            }, '*');
        }
        
        destroy() {
            // Cleanup resources
            if (this.cacheObserver) {
                this.cacheObserver.disconnect();
            }
            
            if (this.performanceInterval) {
                clearInterval(this.performanceInterval);
            }
            
            this.elementCache.clear();
            this.selectorCache.clear();
        }
    }
    
    // Initialize Edge Kernel
    window.SuperOmegaEdgeKernel = new SuperOmegaEdgeKernel();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (window.SuperOmegaEdgeKernel) {
            window.SuperOmegaEdgeKernel.destroy();
        }
    });
    
})();