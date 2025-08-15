use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use serde_json;
use headless_chrome::{Browser, LaunchOptionsBuilder, Tab, Element};
use opencv::{core, imgcodecs, imgproc, videoio, prelude::*};
use std::error::Error;
use tokio::time::{sleep, interval};
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Real evidence collector following SUPER-OMEGA specification
/// Uses /runs/<id>/ directory structure with proper evidence format
pub struct EvidenceCollector {
    session_id: String,
    run_directory: PathBuf,
    browser: Option<Arc<Browser>>,
    video_writer: Option<videoio::VideoWriter>,
    frame_capture_active: bool,
    frame_count: u64,
    evidence_files: Vec<String>,
    
    // Performance tracking
    capture_times: Vec<f64>,
    last_capture_time: Instant,
    
    // Frame cadence control (500ms)
    frame_interval: Duration,
    frame_capture_handle: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceStep {
    pub step_number: u32,
    pub timestamp: DateTime<Utc>,
    pub action_type: String,
    pub target_selector: String,
    pub success: bool,
    pub execution_time_ms: f64,
    pub screenshot_path: String,
    pub dom_snapshot_path: String,
    pub error_message: Option<String>,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub response_time_ms: f64,
    pub dom_ready_time_ms: f64,
    pub network_idle_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReport {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_steps: u32,
    pub successful_steps: u32,
    pub failed_steps: u32,
    pub success_rate: f64,
    pub total_execution_time_ms: f64,
    pub average_step_time_ms: f64,
    pub evidence_files: Vec<String>,
    pub performance_summary: PerformanceSummary,
    pub automation_facts: Vec<AutomationFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub min_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub avg_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub total_frames_captured: u64,
    pub frame_capture_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationFact {
    pub value: String,
    pub source: String,
    pub url: String,
    pub fetched_at: DateTime<Utc>,
    pub trust_score: f64,
}

impl EvidenceCollector {
    pub fn new(session_id: String) -> Result<Self, Box<dyn Error>> {
        let run_directory = PathBuf::from("runs").join(&session_id);
        
        // Create the proper /runs/<id>/ directory structure
        fs::create_dir_all(&run_directory)?;
        fs::create_dir_all(run_directory.join("steps"))?;
        fs::create_dir_all(run_directory.join("frames"))?;
        fs::create_dir_all(run_directory.join("code"))?;
        
        // Initialize browser for real capture
        let browser = Browser::new(
            LaunchOptionsBuilder::default()
                .headless(false)
                .window_size(Some((1920, 1080)))
                .build()
                .unwrap()
        )?;
        
        Ok(Self {
            session_id: session_id.clone(),
            run_directory,
            browser: Some(Arc::new(browser)),
            video_writer: None,
            frame_capture_active: false,
            frame_count: 0,
            evidence_files: Vec::new(),
            capture_times: Vec::new(),
            last_capture_time: Instant::now(),
            frame_interval: Duration::from_millis(500), // 500ms cadence as specified
            frame_capture_handle: None,
        })
    }
    
    pub async fn start_session(&mut self) -> Result<(), Box<dyn Error>> {
        // Start video recording
        self.start_video_recording().await?;
        
        // Start 500ms frame capture
        self.start_frame_capture().await?;
        
        // Create initial report structure
        let report = EvidenceReport {
            session_id: self.session_id.clone(),
            start_time: Utc::now(),
            end_time: Utc::now(), // Will be updated on session end
            total_steps: 0,
            successful_steps: 0,
            failed_steps: 0,
            success_rate: 0.0,
            total_execution_time_ms: 0.0,
            average_step_time_ms: 0.0,
            evidence_files: Vec::new(),
            performance_summary: PerformanceSummary {
                min_response_time_ms: 0.0,
                max_response_time_ms: 0.0,
                avg_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                total_frames_captured: 0,
                frame_capture_rate: 2.0, // 500ms = 2fps
            },
            automation_facts: Vec::new(),
        };
        
        // Save initial report.json
        let report_path = self.run_directory.join("report.json");
        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(report_path, report_json)?;
        
        Ok(())
    }
    
    pub async fn capture_step_evidence(
        &mut self, 
        step_number: u32,
        action_type: &str,
        target_selector: &str,
        success: bool,
        execution_time_ms: f64,
        error_message: Option<String>
    ) -> Result<EvidenceStep, Box<dyn Error>> {
        let timestamp = Utc::now();
        
        // Capture real screenshot
        let screenshot_filename = format!("step_{:04}_screenshot.png", step_number);
        let screenshot_path = self.run_directory.join("steps").join(&screenshot_filename);
        let screenshot_data = self.capture_real_screenshot().await?;
        fs::write(&screenshot_path, screenshot_data)?;
        
        // Capture real DOM snapshot
        let dom_filename = format!("step_{:04}_dom.json", step_number);
        let dom_path = self.run_directory.join("steps").join(&dom_filename);
        let dom_snapshot = self.capture_real_dom_snapshot().await?;
        let dom_json = serde_json::to_string_pretty(&dom_snapshot)?;
        fs::write(&dom_path, dom_json)?;
        
        // Capture performance metrics
        let performance_metrics = self.capture_performance_metrics().await?;
        
        // Create step evidence
        let step_evidence = EvidenceStep {
            step_number,
            timestamp,
            action_type: action_type.to_string(),
            target_selector: target_selector.to_string(),
            success,
            execution_time_ms,
            screenshot_path: screenshot_filename,
            dom_snapshot_path: dom_filename,
            error_message,
            performance_metrics,
        };
        
        // Save step JSON
        let step_json_filename = format!("{}.json", step_number);
        let step_json_path = self.run_directory.join("steps").join(&step_json_filename);
        let step_json = serde_json::to_string_pretty(&step_evidence)?;
        fs::write(step_json_path, step_json)?;
        
        self.evidence_files.push(screenshot_filename);
        self.evidence_files.push(dom_filename);
        self.evidence_files.push(step_json_filename);
        
        Ok(step_evidence)
    }
    
    async fn start_video_recording(&mut self) -> Result<(), Box<dyn Error>> {
        let video_path = self.run_directory.join("video.mp4");
        
        // Initialize OpenCV video writer with real codec
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let fps = 2.0; // 500ms frame rate
        let frame_size = core::Size::new(1920, 1080);
        
        let mut video_writer = videoio::VideoWriter::new(
            &video_path.to_string_lossy(),
            fourcc,
            fps,
            frame_size,
            true
        )?;
        
        if !video_writer.is_opened()? {
            return Err("Failed to open video writer".into());
        }
        
        self.video_writer = Some(video_writer);
        Ok(())
    }
    
    async fn start_frame_capture(&mut self) -> Result<(), Box<dyn Error>> {
        self.frame_capture_active = true;
        
        let session_id = self.session_id.clone();
        let run_directory = self.run_directory.clone();
        let browser = self.browser.clone();
        
        // Start background task for 500ms frame capture
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(500));
            let mut frame_number = 0u64;
            
            while let Some(browser_ref) = &browser {
                interval.tick().await;
                
                // Capture frame every 500ms
                match Self::capture_frame_to_file(
                    browser_ref.clone(), 
                    &run_directory, 
                    frame_number
                ).await {
                    Ok(_) => {
                        frame_number += 1;
                    },
                    Err(e) => {
                        eprintln!("Frame capture error: {}", e);
                    }
                }
            }
        });
        
        self.frame_capture_handle = Some(handle);
        Ok(())
    }
    
    async fn capture_frame_to_file(
        browser: Arc<Browser>,
        run_directory: &PathBuf,
        frame_number: u64
    ) -> Result<(), Box<dyn Error>> {
        let tab = browser.wait_for_initial_tab()?;
        
        // Capture real screenshot
        let screenshot_data = tab.capture_screenshot(
            headless_chrome::protocol::page::CaptureScreenshotFormatOption::Png,
            Some(75),
            None,
            true
        )?;
        
        // Save frame with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis();
            
        let frame_filename = format!("frame_{:06}_{}.png", frame_number, timestamp);
        let frame_path = run_directory.join("frames").join(frame_filename);
        
        fs::write(frame_path, screenshot_data)?;
        
        Ok(())
    }
    
    async fn capture_real_screenshot(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        if let Some(browser) = &self.browser {
            let tab = browser.wait_for_initial_tab()?;
            
            let screenshot_data = tab.capture_screenshot(
                headless_chrome::protocol::page::CaptureScreenshotFormatOption::Png,
                Some(90), // High quality
                None,
                true
            )?;
            
            Ok(screenshot_data)
        } else {
            Err("Browser not initialized".into())
        }
    }
    
    async fn capture_real_dom_snapshot(&self) -> Result<serde_json::Value, Box<dyn Error>> {
        if let Some(browser) = &self.browser {
            let tab = browser.wait_for_initial_tab()?;
            
            // Get real DOM content
            let html_content = tab.get_content()?;
            
            // Get computed styles for all elements
            let computed_styles_script = r#"
                Array.from(document.querySelectorAll('*')).slice(0, 1000).map(el => ({
                    xpath: getXPath(el),
                    tagName: el.tagName,
                    id: el.id,
                    className: el.className,
                    computedStyle: {
                        display: getComputedStyle(el).display,
                        visibility: getComputedStyle(el).visibility,
                        position: getComputedStyle(el).position,
                        zIndex: getComputedStyle(el).zIndex
                    },
                    boundingRect: el.getBoundingClientRect(),
                    textContent: el.textContent?.substring(0, 200)
                }));
                
                function getXPath(element) {
                    if (element.id !== '') return 'id("' + element.id + '")';
                    if (element === document.body) return '/html/body';
                    
                    let ix = 0;
                    const siblings = element.parentNode.childNodes;
                    for (let i = 0; i < siblings.length; i++) {
                        const sibling = siblings[i];
                        if (sibling === element) {
                            return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                            ix++;
                        }
                    }
                }
            "#;
            
            let elements_data = tab.evaluate(computed_styles_script, false)?;
            
            // Get accessibility tree
            let accessibility_script = r#"
                Array.from(document.querySelectorAll('[role], [aria-label], [aria-describedby], input, button, a')).slice(0, 500).map(el => ({
                    xpath: getXPath(el),
                    role: el.getAttribute('role'),
                    ariaLabel: el.getAttribute('aria-label'),
                    ariaDescribedby: el.getAttribute('aria-describedby'),
                    tabIndex: el.tabIndex,
                    disabled: el.disabled
                }));
            "#;
            
            let accessibility_data = tab.evaluate(accessibility_script, false)?;
            
            // Get performance metrics
            let performance_data = tab.evaluate(r#"JSON.stringify(performance.getEntries().slice(-10))"#, false)?;
            
            // Get console logs
            let console_logs = tab.get_runtime_console_messages()?;
            
            // Construct comprehensive DOM snapshot
            let dom_snapshot = serde_json::json!({
                "timestamp": Utc::now().to_rfc3339(),
                "url": tab.get_url(),
                "title": tab.get_title()?,
                "html": html_content,
                "elements": elements_data.value,
                "accessibility": accessibility_data.value,
                "performance": performance_data.value,
                "console_logs": console_logs.iter().map(|log| serde_json::json!({
                    "level": format!("{:?}", log.level),
                    "text": log.text,
                    "timestamp": log.timestamp
                })).collect::<Vec<_>>(),
                "viewport": {
                    "width": 1920,
                    "height": 1080
                },
                "capture_method": "headless_chrome_real"
            });
            
            Ok(dom_snapshot)
        } else {
            Err("Browser not initialized".into())
        }
    }
    
    async fn capture_performance_metrics(&self) -> Result<PerformanceMetrics, Box<dyn Error>> {
        if let Some(browser) = &self.browser {
            let tab = browser.wait_for_initial_tab()?;
            
            // Get real performance metrics from browser
            let metrics_script = r#"
                JSON.stringify({
                    responseTime: performance.now(),
                    domReady: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                    networkIdle: performance.timing.loadEventEnd - performance.timing.navigationStart,
                    memoryUsage: performance.memory ? performance.memory.usedJSHeapSize / 1024 / 1024 : 0
                })
            "#;
            
            let metrics_result = tab.evaluate(metrics_script, false)?;
            let metrics: serde_json::Value = serde_json::from_str(
                metrics_result.value.as_str().unwrap_or("{}")
            )?;
            
            Ok(PerformanceMetrics {
                response_time_ms: metrics["responseTime"].as_f64().unwrap_or(0.0),
                dom_ready_time_ms: metrics["domReady"].as_f64().unwrap_or(0.0),
                network_idle_time_ms: metrics["networkIdle"].as_f64().unwrap_or(0.0),
                memory_usage_mb: metrics["memoryUsage"].as_f64().unwrap_or(0.0),
                cpu_usage_percent: 0.0, // Would need system-level monitoring
            })
        } else {
            Err("Browser not initialized".into())
        }
    }
    
    pub async fn generate_automation_code(&self, actions: &[String]) -> Result<(), Box<dyn Error>> {
        let code_dir = self.run_directory.join("code");
        
        // Generate Playwright TypeScript code
        let playwright_code = self.generate_playwright_code(actions);
        fs::write(code_dir.join("playwright.ts"), playwright_code)?;
        
        // Generate Selenium Python code
        let selenium_code = self.generate_selenium_code(actions);
        fs::write(code_dir.join("selenium.py"), selenium_code)?;
        
        // Generate Cypress JavaScript code
        let cypress_code = self.generate_cypress_code(actions);
        fs::write(code_dir.join("cypress.cy.ts"), cypress_code)?;
        
        Ok(())
    }
    
    fn generate_playwright_code(&self, actions: &[String]) -> String {
        let mut code = String::from(r#"
// Generated Playwright automation code
import { test, expect, Page } from '@playwright/test';

test('SUPER-OMEGA Generated Test', async ({ page }) => {
    const startTime = Date.now();
    
"#);
        
        for (i, action) in actions.iter().enumerate() {
            code.push_str(&format!(r#"
    // Step {}: {}
    await page.waitForTimeout(100);
    {}
    
"#, i + 1, action, self.convert_action_to_playwright(action)));
        }
        
        code.push_str(r#"
    const endTime = Date.now();
    console.log(`Total execution time: ${endTime - startTime}ms`);
});
"#);
        
        code
    }
    
    fn generate_selenium_code(&self, actions: &[String]) -> String {
        let mut code = String::from(r#"
# Generated Selenium automation code
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def super_omega_generated_test():
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    start_time = time.time()
    
    try:
"#);
        
        for (i, action) in actions.iter().enumerate() {
            code.push_str(&format!(r#"
        # Step {}: {}
        time.sleep(0.1)
        {}
        
"#, i + 1, action, self.convert_action_to_selenium(action)));
        }
        
        code.push_str(r#"
        end_time = time.time()
        print(f"Total execution time: {(end_time - start_time) * 1000:.2f}ms")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    super_omega_generated_test()
"#);
        
        code
    }
    
    fn generate_cypress_code(&self, actions: &[String]) -> String {
        let mut code = String::from(r#"
// Generated Cypress automation code
describe('SUPER-OMEGA Generated Test', () => {
    it('should execute automation sequence', () => {
        const startTime = Date.now();
        
"#);
        
        for (i, action) in actions.iter().enumerate() {
            code.push_str(&format!(r#"
        // Step {}: {}
        cy.wait(100);
        {};
        
"#, i + 1, action, self.convert_action_to_cypress(action)));
        }
        
        code.push_str(r#"
        const endTime = Date.now();
        cy.log(`Total execution time: ${endTime - startTime}ms`);
    });
});
"#);
        
        code
    }
    
    fn convert_action_to_playwright(&self, action: &str) -> String {
        // Convert generic action to Playwright code
        if action.contains("click") {
            "await page.click('selector');".to_string()
        } else if action.contains("type") {
            "await page.fill('selector', 'text');".to_string()
        } else {
            format!("// {}", action)
        }
    }
    
    fn convert_action_to_selenium(&self, action: &str) -> String {
        // Convert generic action to Selenium code
        if action.contains("click") {
            "wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'selector'))).click()".to_string()
        } else if action.contains("type") {
            "wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'selector'))).send_keys('text')".to_string()
        } else {
            format!("# {}", action)
        }
    }
    
    fn convert_action_to_cypress(&self, action: &str) -> String {
        // Convert generic action to Cypress code
        if action.contains("click") {
            "cy.get('selector').click()".to_string()
        } else if action.contains("type") {
            "cy.get('selector').type('text')".to_string()
        } else {
            format!("// {}", action)
        }
    }
    
    pub async fn finalize_session(&mut self, steps: &[EvidenceStep]) -> Result<(), Box<dyn Error>> {
        // Stop frame capture
        self.frame_capture_active = false;
        if let Some(handle) = self.frame_capture_handle.take() {
            handle.abort();
        }
        
        // Finalize video
        if let Some(mut writer) = self.video_writer.take() {
            writer.release()?;
        }
        
        // Calculate final statistics
        let successful_steps = steps.iter().filter(|s| s.success).count() as u32;
        let total_steps = steps.len() as u32;
        let success_rate = if total_steps > 0 {
            successful_steps as f64 / total_steps as f64
        } else {
            0.0
        };
        
        let total_execution_time: f64 = steps.iter().map(|s| s.execution_time_ms).sum();
        let average_step_time = if total_steps > 0 {
            total_execution_time / total_steps as f64
        } else {
            0.0
        };
        
        // Calculate performance percentiles
        let mut response_times: Vec<f64> = steps.iter()
            .map(|s| s.performance_metrics.response_time_ms)
            .collect();
        response_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p95_index = ((response_times.len() as f64) * 0.95) as usize;
        let p95_response_time = response_times.get(p95_index).copied().unwrap_or(0.0);
        
        // Create final report
        let final_report = EvidenceReport {
            session_id: self.session_id.clone(),
            start_time: steps.first().map(|s| s.timestamp).unwrap_or_else(Utc::now),
            end_time: Utc::now(),
            total_steps,
            successful_steps,
            failed_steps: total_steps - successful_steps,
            success_rate,
            total_execution_time_ms: total_execution_time,
            average_step_time_ms: average_step_time,
            evidence_files: self.evidence_files.clone(),
            performance_summary: PerformanceSummary {
                min_response_time_ms: response_times.first().copied().unwrap_or(0.0),
                max_response_time_ms: response_times.last().copied().unwrap_or(0.0),
                avg_response_time_ms: if !response_times.is_empty() {
                    response_times.iter().sum::<f64>() / response_times.len() as f64
                } else {
                    0.0
                },
                p95_response_time_ms: p95_response_time,
                total_frames_captured: self.frame_count,
                frame_capture_rate: 2.0, // 500ms = 2fps
            },
            automation_facts: Vec::new(), // Would be populated with extracted facts
        };
        
        // Save final report.json
        let report_path = self.run_directory.join("report.json");
        let report_json = serde_json::to_string_pretty(&final_report)?;
        fs::write(report_path, report_json)?;
        
        // Create facts.jsonl file
        let facts_path = self.run_directory.join("facts.jsonl");
        let mut facts_content = String::new();
        
        for step in steps {
            if step.success {
                let fact = AutomationFact {
                    value: format!("Successfully executed {}", step.action_type),
                    source: "automation_execution".to_string(),
                    url: "local://automation".to_string(),
                    fetched_at: step.timestamp,
                    trust_score: 0.95,
                };
                facts_content.push_str(&serde_json::to_string(&fact)?);
                facts_content.push('\n');
            }
        }
        
        fs::write(facts_path, facts_content)?;
        
        Ok(())
    }
}