use tauri::{Manager, State};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use dashmap::DashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use log::{info, warn, error};

mod micro_planner;
mod dom_capture;
mod action_executor;
mod vision_processor;
mod performance_monitor;
mod evidence_collector;

use micro_planner::MicroPlanner;
use dom_capture::DOMCapture;
use action_executor::ActionExecutor;
use vision_processor::VisionProcessor;
use performance_monitor::PerformanceMonitor;
use evidence_collector::EvidenceCollector;

/// Edge Kernel State - Ultra-fast automation engine
#[derive(Debug)]
pub struct EdgeKernel {
    micro_planner: Arc<MicroPlanner>,
    dom_capture: Arc<DOMCapture>,
    action_executor: Arc<ActionExecutor>,
    vision_processor: Arc<VisionProcessor>,
    performance_monitor: Arc<PerformanceMonitor>,
    evidence_collector: Arc<EvidenceCollector>,
    active_sessions: Arc<DashMap<String, AutomationSession>>,
    performance_cache: Arc<RwLock<DashMap<String, CachedAction>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationSession {
    pub id: String,
    pub start_time: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub tab_id: Option<u32>,
    pub current_url: String,
    pub performance_metrics: PerformanceMetrics,
    pub evidence_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_actions: u64,
    pub successful_actions: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub healing_events: u64,
    pub sub_25ms_actions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAction {
    pub selector: String,
    pub action_type: String,
    pub success_count: u64,
    pub failure_count: u64,
    pub average_latency_ms: f64,
    pub last_success: DateTime<Utc>,
    pub visual_fingerprint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DOMSnapshot {
    pub html: String,
    pub accessibility_tree: serde_json::Value,
    pub computed_styles: serde_json::Value,
    pub screenshot_base64: String,
    pub viewport: Viewport,
    pub timestamp: DateTime<Utc>,
    pub performance_timing: PerformanceTiming,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f64,
    pub is_mobile: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceTiming {
    pub dom_capture_ms: f64,
    pub accessibility_parse_ms: f64,
    pub screenshot_capture_ms: f64,
    pub total_snapshot_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionRequest {
    pub session_id: String,
    pub action_type: String,
    pub target: ActionTarget,
    pub value: Option<String>,
    pub timeout_ms: u64,
    pub retry_count: u8,
    pub evidence_level: EvidenceLevel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionTarget {
    pub role: Option<String>,
    pub name: Option<String>,
    pub text: Option<String>,
    pub css_selector: Option<String>,
    pub xpath: Option<String>,
    pub visual_template: Option<String>,
    pub coordinates: Option<(f64, f64)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EvidenceLevel {
    Minimal,    // Basic timing and success/failure
    Standard,   // + screenshots before/after
    Full,       // + DOM snapshots, video recording
    Forensic,   // + network logs, console logs, performance traces
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    pub execution_time_ms: f64,
    pub selector_used: String,
    pub healing_applied: bool,
    pub evidence_files: Vec<String>,
    pub error_message: Option<String>,
    pub performance_data: ActionPerformanceData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionPerformanceData {
    pub element_location_ms: f64,
    pub action_execution_ms: f64,
    pub verification_ms: f64,
    pub total_ms: f64,
    pub cache_hit: bool,
    pub healing_attempts: u8,
}

impl EdgeKernel {
    pub async fn new() -> Result<Self> {
        info!("Initializing SUPER-OMEGA Edge Kernel...");
        
        let micro_planner = Arc::new(MicroPlanner::new().await?);
        let dom_capture = Arc::new(DOMCapture::new().await?);
        let action_executor = Arc::new(ActionExecutor::new().await?);
        let vision_processor = Arc::new(VisionProcessor::new().await?);
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        let evidence_collector = Arc::new(EvidenceCollector::new().await?);
        
        Ok(Self {
            micro_planner,
            dom_capture,
            action_executor,
            vision_processor,
            performance_monitor,
            evidence_collector,
            active_sessions: Arc::new(DashMap::new()),
            performance_cache: Arc::new(RwLock::new(DashMap::new())),
        })
    }
    
    /// Create new automation session with sub-25ms initialization
    pub async fn create_session(&self, url: Option<String>) -> Result<String> {
        let start_time = Instant::now();
        let session_id = Uuid::new_v4().to_string();
        
        let session = AutomationSession {
            id: session_id.clone(),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            tab_id: None,
            current_url: url.unwrap_or_default(),
            performance_metrics: PerformanceMetrics {
                total_actions: 0,
                successful_actions: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                cache_hit_rate: 0.0,
                healing_events: 0,
                sub_25ms_actions: 0,
            },
            evidence_path: format!("runs/{}/", session_id),
        };
        
        // Create evidence directory structure immediately
        self.evidence_collector.create_session_structure(&session_id).await?;
        
        self.active_sessions.insert(session_id.clone(), session);
        
        let initialization_time = start_time.elapsed().as_millis();
        info!("Session {} created in {}ms", session_id, initialization_time);
        
        // Target: Session creation should be sub-5ms
        if initialization_time > 5 {
            warn!("Session creation exceeded 5ms target: {}ms", initialization_time);
        }
        
        Ok(session_id)
    }
    
    /// Get DOM snapshot with sub-10ms target
    pub async fn get_dom_snapshot(&self, session_id: &str) -> Result<DOMSnapshot> {
        let start_time = Instant::now();
        
        let session = self.active_sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        let snapshot = self.dom_capture.capture_full_snapshot(&session.current_url).await?;
        
        let capture_time = start_time.elapsed().as_millis() as f64;
        
        // Update session activity
        if let Some(mut session) = self.active_sessions.get_mut(session_id) {
            session.last_activity = Utc::now();
        }
        
        // Performance monitoring
        self.performance_monitor.record_dom_capture(capture_time).await;
        
        if capture_time > 10.0 {
            warn!("DOM capture exceeded 10ms target: {:.2}ms", capture_time);
        }
        
        Ok(snapshot)
    }
    
    /// Execute action with sub-25ms target (the core requirement)
    pub async fn perform_action(&self, request: ActionRequest) -> Result<ActionResult> {
        let execution_start = Instant::now();
        
        // Validate session
        let session = self.active_sessions.get(&request.session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", request.session_id))?;
        
        // Check performance cache first (sub-1ms cache lookup)
        let cache_key = format!("{}:{}:{:?}", 
            request.action_type, 
            request.target.css_selector.as_deref().unwrap_or(""), 
            request.target.role
        );
        
        let cache_hit = {
            let cache = self.performance_cache.read().await;
            cache.contains_key(&cache_key)
        };
        
        // Execute action with micro-planner optimization
        let result = if cache_hit {
            // Ultra-fast cached execution path
            self.execute_cached_action(&request, &cache_key).await?
        } else {
            // Full execution with caching
            self.execute_full_action(&request, &cache_key).await?
        };
        
        let total_execution_time = execution_start.elapsed().as_millis() as f64;
        
        // Update performance metrics
        self.update_session_metrics(&request.session_id, total_execution_time, result.success).await?;
        
        // Record evidence
        self.evidence_collector.record_action_evidence(
            &request.session_id,
            &request,
            &result,
            request.evidence_level
        ).await?;
        
        // Performance validation - CRITICAL REQUIREMENT
        if total_execution_time <= 25.0 {
            if let Some(mut session) = self.active_sessions.get_mut(&request.session_id) {
                session.performance_metrics.sub_25ms_actions += 1;
            }
            info!("✅ Sub-25ms action achieved: {:.2}ms", total_execution_time);
        } else {
            warn!("❌ Action exceeded 25ms target: {:.2}ms", total_execution_time);
        }
        
        Ok(result)
    }
    
    /// Execute cached action (target: sub-5ms)
    async fn execute_cached_action(&self, request: &ActionRequest, cache_key: &str) -> Result<ActionResult> {
        let start_time = Instant::now();
        
        let cached_action = {
            let cache = self.performance_cache.read().await;
            cache.get(cache_key).cloned()
        };
        
        if let Some(cached) = cached_action {
            let result = self.action_executor.execute_cached(
                &request.target,
                &request.action_type,
                request.value.as_deref(),
                &cached.selector
            ).await?;
            
            let execution_time = start_time.elapsed().as_millis() as f64;
            
            Ok(ActionResult {
                success: result.success,
                execution_time_ms: execution_time,
                selector_used: cached.selector,
                healing_applied: false,
                evidence_files: vec![],
                error_message: result.error_message,
                performance_data: ActionPerformanceData {
                    element_location_ms: 0.0, // Cached
                    action_execution_ms: execution_time,
                    verification_ms: 0.0,
                    total_ms: execution_time,
                    cache_hit: true,
                    healing_attempts: 0,
                },
            })
        } else {
            // Cache miss - fall back to full execution
            self.execute_full_action(request, cache_key).await
        }
    }
    
    /// Execute full action with element location and healing
    async fn execute_full_action(&self, request: &ActionRequest, cache_key: &str) -> Result<ActionResult> {
        let location_start = Instant::now();
        
        // Use micro-planner for optimal selector strategy
        let selector_strategy = self.micro_planner.plan_selector_strategy(&request.target).await?;
        
        // Execute with self-healing fallbacks
        let mut healing_attempts = 0;
        let mut last_error = None;
        
        for selector in selector_strategy.selectors {
            let attempt_start = Instant::now();
            
            match self.action_executor.execute_with_selector(
                &selector,
                &request.action_type,
                request.value.as_deref(),
                request.timeout_ms
            ).await {
                Ok(result) => {
                    let location_time = location_start.elapsed().as_millis() as f64;
                    let execution_time = attempt_start.elapsed().as_millis() as f64;
                    let total_time = location_start.elapsed().as_millis() as f64;
                    
                    // Cache successful action
                    self.cache_successful_action(cache_key, &selector, total_time).await;
                    
                    return Ok(ActionResult {
                        success: true,
                        execution_time_ms: total_time,
                        selector_used: selector,
                        healing_applied: healing_attempts > 0,
                        evidence_files: vec![],
                        error_message: None,
                        performance_data: ActionPerformanceData {
                            element_location_ms: location_time - execution_time,
                            action_execution_ms: execution_time,
                            verification_ms: 0.0,
                            total_ms: total_time,
                            cache_hit: false,
                            healing_attempts,
                        },
                    });
                }
                Err(e) => {
                    healing_attempts += 1;
                    last_error = Some(e);
                    if healing_attempts >= 3 {
                        break;
                    }
                }
            }
        }
        
        let total_time = location_start.elapsed().as_millis() as f64;
        
        Ok(ActionResult {
            success: false,
            execution_time_ms: total_time,
            selector_used: "none".to_string(),
            healing_applied: healing_attempts > 0,
            evidence_files: vec![],
            error_message: last_error.map(|e| e.to_string()),
            performance_data: ActionPerformanceData {
                element_location_ms: total_time,
                action_execution_ms: 0.0,
                verification_ms: 0.0,
                total_ms: total_time,
                cache_hit: false,
                healing_attempts,
            },
        })
    }
    
    /// Cache successful action for future ultra-fast execution
    async fn cache_successful_action(&self, cache_key: &str, selector: &str, execution_time: f64) {
        let cached_action = CachedAction {
            selector: selector.to_string(),
            action_type: "cached".to_string(),
            success_count: 1,
            failure_count: 0,
            average_latency_ms: execution_time,
            last_success: Utc::now(),
            visual_fingerprint: None,
        };
        
        let mut cache = self.performance_cache.write().await;
        cache.insert(cache_key.to_string(), cached_action);
    }
    
    /// Update session performance metrics
    async fn update_session_metrics(&self, session_id: &str, execution_time: f64, success: bool) -> Result<()> {
        if let Some(mut session) = self.active_sessions.get_mut(session_id) {
            session.performance_metrics.total_actions += 1;
            if success {
                session.performance_metrics.successful_actions += 1;
            }
            
            // Update running average
            let total = session.performance_metrics.total_actions as f64;
            let current_avg = session.performance_metrics.average_latency_ms;
            session.performance_metrics.average_latency_ms = 
                (current_avg * (total - 1.0) + execution_time) / total;
            
            session.last_activity = Utc::now();
        }
        
        Ok(())
    }
    
    /// Get performance statistics
    pub async fn get_performance_stats(&self, session_id: &str) -> Result<PerformanceMetrics> {
        let session = self.active_sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        Ok(session.performance_metrics.clone())
    }
    
    /// Start video recording for evidence
    pub async fn start_video_recording(&self, session_id: &str) -> Result<()> {
        self.evidence_collector.start_video_recording(session_id).await
    }
    
    /// Stop video recording
    pub async fn stop_video_recording(&self, session_id: &str) -> Result<String> {
        self.evidence_collector.stop_video_recording(session_id).await
    }
}

/// Tauri command handlers
#[tauri::command]
async fn create_session(
    state: State<'_, Arc<EdgeKernel>>,
    url: Option<String>
) -> Result<String, String> {
    state.create_session(url).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_dom_snapshot(
    state: State<'_, Arc<EdgeKernel>>,
    session_id: String
) -> Result<DOMSnapshot, String> {
    state.get_dom_snapshot(&session_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn perform_action(
    state: State<'_, Arc<EdgeKernel>>,
    request: ActionRequest
) -> Result<ActionResult, String> {
    state.perform_action(request).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_performance_stats(
    state: State<'_, Arc<EdgeKernel>>,
    session_id: String
) -> Result<PerformanceMetrics, String> {
    state.get_performance_stats(&session_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn start_video_recording(
    state: State<'_, Arc<EdgeKernel>>,
    session_id: String
) -> Result<(), String> {
    state.start_video_recording(&session_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn stop_video_recording(
    state: State<'_, Arc<EdgeKernel>>,
    session_id: String
) -> Result<String, String> {
    state.stop_video_recording(&session_id).await.map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    info!("Starting SUPER-OMEGA Edge Kernel...");
    
    let edge_kernel = Arc::new(EdgeKernel::new().await?);
    
    tauri::Builder::default()
        .manage(edge_kernel)
        .invoke_handler(tauri::generate_handler![
            create_session,
            get_dom_snapshot,
            perform_action,
            get_performance_stats,
            start_video_recording,
            stop_video_recording
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
    
    Ok(())
}