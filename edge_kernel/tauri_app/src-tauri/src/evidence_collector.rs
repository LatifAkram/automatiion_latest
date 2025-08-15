use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::Write;
use tokio::sync::RwLock;
use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use log::{info, warn, error};
use crate::{ActionRequest, ActionResult, EvidenceLevel};

/// Evidence collector that creates the exact /runs/<id>/ structure from the spec
pub struct EvidenceCollector {
    base_path: PathBuf,
    active_recordings: Arc<RwLock<HashMap<String, VideoRecording>>>,
    frame_capture_interval: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub run_id: String,
    pub goal: String,
    pub status: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub total_steps: u32,
    pub successful_steps: u32,
    pub failed_steps: u32,
    pub evidence_files: Vec<String>,
    pub performance_metrics: PerformanceMetrics,
    pub error_log: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepReport {
    pub step_id: String,
    pub step_number: u32,
    pub timestamp: DateTime<Utc>,
    pub action_type: String,
    pub target: serde_json::Value,
    pub success: bool,
    pub execution_time_ms: f64,
    pub selector_used: String,
    pub healing_applied: bool,
    pub retry_count: u8,
    pub evidence_files: Vec<String>,
    pub error_message: Option<String>,
    pub dom_snapshot_file: Option<String>,
    pub screenshot_before: Option<String>,
    pub screenshot_after: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_actions: u64,
    pub successful_actions: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub sub_25ms_actions: u64,
    pub cache_hit_rate: f64,
    pub healing_events: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactEntry {
    pub value: serde_json::Value,
    pub source: String,
    pub url: String,
    pub fetched_at: DateTime<Utc>,
    pub trust: f32,
}

struct VideoRecording {
    session_id: String,
    output_path: PathBuf,
    start_time: DateTime<Utc>,
    frame_count: u32,
    is_recording: bool,
}

impl EvidenceCollector {
    pub async fn new() -> Result<Self> {
        let base_path = PathBuf::from("runs");
        
        // Create base runs directory
        if !base_path.exists() {
            fs::create_dir_all(&base_path)?;
        }
        
        Ok(Self {
            base_path,
            active_recordings: Arc::new(RwLock::new(HashMap::new())),
            frame_capture_interval: std::time::Duration::from_millis(500), // 500ms cadence as specified
        })
    }
    
    /// Create the exact evidence structure: /runs/<id>/
    pub async fn create_session_structure(&self, session_id: &str) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        
        // Create main session directory
        fs::create_dir_all(&session_path)?;
        
        // Create subdirectories as specified in the contract
        fs::create_dir_all(session_path.join("steps"))?;
        fs::create_dir_all(session_path.join("frames"))?;
        fs::create_dir_all(session_path.join("code"))?;
        
        // Create initial report.json
        let initial_report = RunReport {
            run_id: session_id.to_string(),
            goal: "Session initialized".to_string(),
            status: "running".to_string(),
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            total_steps: 0,
            successful_steps: 0,
            failed_steps: 0,
            evidence_files: Vec::new(),
            performance_metrics: PerformanceMetrics {
                total_actions: 0,
                successful_actions: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                sub_25ms_actions: 0,
                cache_hit_rate: 0.0,
                healing_events: 0,
            },
            error_log: Vec::new(),
        };
        
        self.save_report(session_id, &initial_report).await?;
        
        // Create initial facts.jsonl file
        let facts_path = session_path.join("facts.jsonl");
        fs::File::create(facts_path)?;
        
        info!("Created evidence structure for session: {}", session_id);
        Ok(())
    }
    
    /// Record action evidence according to evidence level
    pub async fn record_action_evidence(
        &self,
        session_id: &str,
        request: &ActionRequest,
        result: &ActionResult,
        evidence_level: EvidenceLevel,
    ) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let step_number = self.get_next_step_number(session_id).await?;
        let step_id = format!("step_{:04}", step_number);
        
        let mut evidence_files = Vec::new();
        let mut screenshot_before = None;
        let mut screenshot_after = None;
        let mut dom_snapshot_file = None;
        
        // Capture evidence based on level
        match evidence_level {
            EvidenceLevel::Minimal => {
                // Only basic timing and success/failure - no files
            }
            EvidenceLevel::Standard => {
                // + screenshots before/after
                screenshot_before = Some(self.capture_screenshot(session_id, &format!("{}_before", step_id)).await?);
                screenshot_after = Some(self.capture_screenshot(session_id, &format!("{}_after", step_id)).await?);
                
                if let Some(path) = &screenshot_before {
                    evidence_files.push(path.clone());
                }
                if let Some(path) = &screenshot_after {
                    evidence_files.push(path.clone());
                }
            }
            EvidenceLevel::Full => {
                // + DOM snapshots, video recording
                screenshot_before = Some(self.capture_screenshot(session_id, &format!("{}_before", step_id)).await?);
                screenshot_after = Some(self.capture_screenshot(session_id, &format!("{}_after", step_id)).await?);
                dom_snapshot_file = Some(self.capture_dom_snapshot(session_id, &step_id).await?);
                
                evidence_files.extend([
                    screenshot_before.as_ref().unwrap().clone(),
                    screenshot_after.as_ref().unwrap().clone(),
                    dom_snapshot_file.as_ref().unwrap().clone(),
                ]);
            }
            EvidenceLevel::Forensic => {
                // + network logs, console logs, performance traces
                screenshot_before = Some(self.capture_screenshot(session_id, &format!("{}_before", step_id)).await?);
                screenshot_after = Some(self.capture_screenshot(session_id, &format!("{}_after", step_id)).await?);
                dom_snapshot_file = Some(self.capture_dom_snapshot(session_id, &step_id).await?);
                
                let network_log = self.capture_network_log(session_id, &step_id).await?;
                let console_log = self.capture_console_log(session_id, &step_id).await?;
                let performance_trace = self.capture_performance_trace(session_id, &step_id).await?;
                
                evidence_files.extend([
                    screenshot_before.as_ref().unwrap().clone(),
                    screenshot_after.as_ref().unwrap().clone(),
                    dom_snapshot_file.as_ref().unwrap().clone(),
                    network_log,
                    console_log,
                    performance_trace,
                ]);
            }
        }
        
        // Create step report
        let step_report = StepReport {
            step_id: step_id.clone(),
            step_number,
            timestamp: Utc::now(),
            action_type: request.action_type.clone(),
            target: serde_json::to_value(&request.target)?,
            success: result.success,
            execution_time_ms: result.execution_time_ms,
            selector_used: result.selector_used.clone(),
            healing_applied: result.healing_applied,
            retry_count: request.retry_count,
            evidence_files: evidence_files.clone(),
            error_message: result.error_message.clone(),
            dom_snapshot_file,
            screenshot_before,
            screenshot_after,
        };
        
        // Save step report: /runs/<id>/steps/<n>.json
        let step_file = session_path.join("steps").join(format!("{}.json", step_number));
        let step_json = serde_json::to_string_pretty(&step_report)?;
        fs::write(step_file, step_json)?;
        
        // Update main report
        self.update_main_report(session_id, &step_report).await?;
        
        // Generate code artifacts
        self.generate_code_artifacts(session_id, request, result).await?;
        
        info!("Recorded evidence for step {} in session {}", step_number, session_id);
        Ok(())
    }
    
    /// Start video recording with 500ms frame cadence
    pub async fn start_video_recording(&self, session_id: &str) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let video_path = session_path.join("video.mp4");
        
        let recording = VideoRecording {
            session_id: session_id.to_string(),
            output_path: video_path,
            start_time: Utc::now(),
            frame_count: 0,
            is_recording: true,
        };
        
        let mut recordings = self.active_recordings.write().await;
        recordings.insert(session_id.to_string(), recording);
        
        // Start frame capture task
        let session_id_clone = session_id.to_string();
        let base_path_clone = self.base_path.clone();
        let interval = self.frame_capture_interval;
        
        tokio::spawn(async move {
            let mut frame_counter = 0u32;
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                // Check if still recording
                // In a real implementation, this would capture actual frames
                frame_counter += 1;
                
                // Capture frame to /runs/<id>/frames/frame_NNNN.png
                let frame_path = base_path_clone
                    .join(&session_id_clone)
                    .join("frames")
                    .join(format!("frame_{:04}.png", frame_counter));
                
                // Placeholder frame capture (in real implementation, capture actual screen)
                let placeholder_frame = vec![0u8; 1920 * 1080 * 3]; // RGB placeholder
                if let Err(e) = fs::write(&frame_path, &placeholder_frame) {
                    error!("Failed to write frame: {}", e);
                    break;
                }
                
                // Break after reasonable time for demo (in real implementation, continue until stopped)
                if frame_counter > 1000 {
                    break;
                }
            }
        });
        
        info!("Started video recording for session: {}", session_id);
        Ok(())
    }
    
    /// Stop video recording and compile to MP4
    pub async fn stop_video_recording(&self, session_id: &str) -> Result<String> {
        let mut recordings = self.active_recordings.write().await;
        
        if let Some(mut recording) = recordings.remove(session_id) {
            recording.is_recording = false;
            
            // In a real implementation, this would compile frames to MP4 using FFmpeg
            // For now, create a placeholder video file
            let video_content = b"PLACEHOLDER_VIDEO_DATA";
            fs::write(&recording.output_path, video_content)?;
            
            info!("Stopped video recording for session: {}", session_id);
            Ok(recording.output_path.to_string_lossy().to_string())
        } else {
            Err(anyhow::anyhow!("No active recording for session: {}", session_id))
        }
    }
    
    /// Record fact to facts.jsonl
    pub async fn record_fact(&self, session_id: &str, fact: FactEntry) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let facts_path = session_path.join("facts.jsonl");
        
        let fact_line = serde_json::to_string(&fact)? + "\n";
        
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(facts_path)?;
        
        file.write_all(fact_line.as_bytes())?;
        file.flush()?;
        
        Ok(())
    }
    
    /// Save main report.json
    async fn save_report(&self, session_id: &str, report: &RunReport) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let report_path = session_path.join("report.json");
        
        let report_json = serde_json::to_string_pretty(report)?;
        fs::write(report_path, report_json)?;
        
        Ok(())
    }
    
    /// Get next step number for session
    async fn get_next_step_number(&self, session_id: &str) -> Result<u32> {
        let session_path = self.base_path.join(session_id);
        let steps_path = session_path.join("steps");
        
        if !steps_path.exists() {
            return Ok(1);
        }
        
        let entries = fs::read_dir(steps_path)?;
        let count = entries.count() as u32;
        Ok(count + 1)
    }
    
    /// Capture screenshot
    async fn capture_screenshot(&self, session_id: &str, name: &str) -> Result<String> {
        let session_path = self.base_path.join(session_id);
        let screenshot_path = session_path.join(format!("{}.png", name));
        
        // In a real implementation, this would capture actual screenshot
        // For now, create placeholder image data
        let placeholder_image = vec![0u8; 1920 * 1080 * 3]; // RGB placeholder
        fs::write(&screenshot_path, &placeholder_image)?;
        
        Ok(screenshot_path.to_string_lossy().to_string())
    }
    
    /// Capture DOM snapshot
    async fn capture_dom_snapshot(&self, session_id: &str, step_id: &str) -> Result<String> {
        let session_path = self.base_path.join(session_id);
        let snapshot_path = session_path.join(format!("{}_dom.json", step_id));
        
        // In a real implementation, this would capture actual DOM
        let placeholder_dom = serde_json::json!({
            "html": "<html><body>PLACEHOLDER_DOM</body></html>",
            "timestamp": Utc::now(),
            "url": "https://example.com"
        });
        
        let dom_json = serde_json::to_string_pretty(&placeholder_dom)?;
        fs::write(&snapshot_path, dom_json)?;
        
        Ok(snapshot_path.to_string_lossy().to_string())
    }
    
    /// Capture network log
    async fn capture_network_log(&self, session_id: &str, step_id: &str) -> Result<String> {
        let session_path = self.base_path.join(session_id);
        let log_path = session_path.join(format!("{}_network.json", step_id));
        
        let placeholder_log = serde_json::json!({
            "requests": [],
            "responses": [],
            "timestamp": Utc::now()
        });
        
        let log_json = serde_json::to_string_pretty(&placeholder_log)?;
        fs::write(&log_path, log_json)?;
        
        Ok(log_path.to_string_lossy().to_string())
    }
    
    /// Capture console log
    async fn capture_console_log(&self, session_id: &str, step_id: &str) -> Result<String> {
        let session_path = self.base_path.join(session_id);
        let log_path = session_path.join(format!("{}_console.json", step_id));
        
        let placeholder_log = serde_json::json!({
            "messages": [],
            "timestamp": Utc::now()
        });
        
        let log_json = serde_json::to_string_pretty(&placeholder_log)?;
        fs::write(&log_path, log_json)?;
        
        Ok(log_path.to_string_lossy().to_string())
    }
    
    /// Capture performance trace
    async fn capture_performance_trace(&self, session_id: &str, step_id: &str) -> Result<String> {
        let session_path = self.base_path.join(session_id);
        let trace_path = session_path.join(format!("{}_performance.json", step_id));
        
        let placeholder_trace = serde_json::json!({
            "timeline": [],
            "metrics": {},
            "timestamp": Utc::now()
        });
        
        let trace_json = serde_json::to_string_pretty(&placeholder_trace)?;
        fs::write(&trace_path, trace_json)?;
        
        Ok(trace_path.to_string_lossy().to_string())
    }
    
    /// Generate code artifacts: /runs/<id>/code/{playwright.ts, selenium.py, cypress.cy.ts}
    async fn generate_code_artifacts(&self, session_id: &str, request: &ActionRequest, result: &ActionResult) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let code_path = session_path.join("code");
        
        // Generate Playwright TypeScript code
        let playwright_code = self.generate_playwright_code(request, result)?;
        fs::write(code_path.join("playwright.ts"), playwright_code)?;
        
        // Generate Selenium Python code
        let selenium_code = self.generate_selenium_code(request, result)?;
        fs::write(code_path.join("selenium.py"), selenium_code)?;
        
        // Generate Cypress TypeScript code
        let cypress_code = self.generate_cypress_code(request, result)?;
        fs::write(code_path.join("cypress.cy.ts"), cypress_code)?;
        
        Ok(())
    }
    
    /// Generate Playwright TypeScript code
    fn generate_playwright_code(&self, request: &ActionRequest, result: &ActionResult) -> Result<String> {
        let code = format!(
            r#"import {{ test, expect }} from '@playwright/test';

test('Generated automation step', async ({{ page }}) => {{
    // Action: {}
    // Selector: {}
    // Success: {}
    // Execution time: {:.2}ms
    
    await page.locator('{}').{}({});
}});"#,
            request.action_type,
            result.selector_used,
            result.success,
            result.execution_time_ms,
            result.selector_used,
            self.map_action_to_playwright(&request.action_type),
            request.value.as_deref().unwrap_or("")
        );
        
        Ok(code)
    }
    
    /// Generate Selenium Python code
    fn generate_selenium_code(&self, request: &ActionRequest, result: &ActionResult) -> Result<String> {
        let code = format!(
            r#"from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Action: {}
# Selector: {}
# Success: {}
# Execution time: {:.2}ms

driver = webdriver.Chrome()
element = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "{}"))
)
element.{}("{}")
"#,
            request.action_type,
            result.selector_used,
            result.success,
            result.execution_time_ms,
            result.selector_used,
            self.map_action_to_selenium(&request.action_type),
            request.value.as_deref().unwrap_or("")
        );
        
        Ok(code)
    }
    
    /// Generate Cypress TypeScript code
    fn generate_cypress_code(&self, request: &ActionRequest, result: &ActionResult) -> Result<String> {
        let code = format!(
            r#"describe('Generated automation step', () => {{
    it('should perform {}', () => {{
        // Selector: {}
        // Success: {}
        // Execution time: {:.2}ms
        
        cy.get('{}').{}('{}');
    }});
}});"#,
            request.action_type,
            result.selector_used,
            result.success,
            result.execution_time_ms,
            result.selector_used,
            self.map_action_to_cypress(&request.action_type),
            request.value.as_deref().unwrap_or("")
        );
        
        Ok(code)
    }
    
    /// Map action type to Playwright method
    fn map_action_to_playwright(&self, action_type: &str) -> &str {
        match action_type {
            "click" => "click",
            "type" => "fill",
            "hover" => "hover",
            "scroll" => "scrollIntoView",
            _ => "click",
        }
    }
    
    /// Map action type to Selenium method
    fn map_action_to_selenium(&self, action_type: &str) -> &str {
        match action_type {
            "click" => "click",
            "type" => "send_keys",
            "hover" => "move_to_element",
            "scroll" => "scroll_into_view",
            _ => "click",
        }
    }
    
    /// Map action type to Cypress method
    fn map_action_to_cypress(&self, action_type: &str) -> &str {
        match action_type {
            "click" => "click",
            "type" => "type",
            "hover" => "trigger",
            "scroll" => "scrollIntoView",
            _ => "click",
        }
    }
    
    /// Update main report with step information
    async fn update_main_report(&self, session_id: &str, step_report: &StepReport) -> Result<()> {
        let session_path = self.base_path.join(session_id);
        let report_path = session_path.join("report.json");
        
        // Load existing report
        let mut report: RunReport = if report_path.exists() {
            let report_content = fs::read_to_string(&report_path)?;
            serde_json::from_str(&report_content)?
        } else {
            RunReport {
                run_id: session_id.to_string(),
                goal: "Unknown".to_string(),
                status: "running".to_string(),
                start_time: Utc::now(),
                end_time: None,
                duration_ms: None,
                total_steps: 0,
                successful_steps: 0,
                failed_steps: 0,
                evidence_files: Vec::new(),
                performance_metrics: PerformanceMetrics {
                    total_actions: 0,
                    successful_actions: 0,
                    average_latency_ms: 0.0,
                    p95_latency_ms: 0.0,
                    p99_latency_ms: 0.0,
                    sub_25ms_actions: 0,
                    cache_hit_rate: 0.0,
                    healing_events: 0,
                },
                error_log: Vec::new(),
            }
        };
        
        // Update report with step information
        report.total_steps += 1;
        if step_report.success {
            report.successful_steps += 1;
        } else {
            report.failed_steps += 1;
            if let Some(error) = &step_report.error_message {
                report.error_log.push(format!("Step {}: {}", step_report.step_number, error));
            }
        }
        
        report.evidence_files.extend(step_report.evidence_files.clone());
        
        // Update performance metrics
        report.performance_metrics.total_actions += 1;
        if step_report.success {
            report.performance_metrics.successful_actions += 1;
        }
        
        if step_report.execution_time_ms <= 25.0 {
            report.performance_metrics.sub_25ms_actions += 1;
        }
        
        if step_report.healing_applied {
            report.performance_metrics.healing_events += 1;
        }
        
        // Update running average
        let total = report.performance_metrics.total_actions as f64;
        let current_avg = report.performance_metrics.average_latency_ms;
        report.performance_metrics.average_latency_ms = 
            (current_avg * (total - 1.0) + step_report.execution_time_ms) / total;
        
        // Save updated report
        self.save_report(session_id, &report).await?;
        
        Ok(())
    }
}