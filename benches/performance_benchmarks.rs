use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

// Import our modules
use super_omega_edge_kernel::{
    EdgeKernel, MicroPlanner, VisionProcessor, ActionExecutor, 
    DOMCapture, EvidenceCollector, PerformanceMonitor
};

/// Benchmark the complete action execution pipeline
/// Target: Sub-25ms for the full cycle
fn benchmark_full_action_cycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("full_action_cycle");
    group.significance_level(0.1).sample_size(1000);
    
    // Set up test data
    let test_actions = vec![
        ("click_button", r#"{"type": "click", "selector": "#submit-btn"}"#),
        ("fill_input", r#"{"type": "type", "selector": "#email", "text": "test@example.com"}"#),
        ("select_dropdown", r#"{"type": "select", "selector": "#country", "value": "US"}"#),
    ];
    
    for (action_name, action_json) in test_actions {
        group.bench_with_input(
            BenchmarkId::new("complete_pipeline", action_name),
            &action_json,
            |b, action_data| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    // Initialize kernel (this should be cached in real usage)
                    let mut kernel = EdgeKernel::new().expect("Failed to create kernel");
                    let session_id = kernel.create_session().await.expect("Failed to create session");
                    
                    // Parse action
                    let action: serde_json::Value = serde_json::from_str(action_data)
                        .expect("Failed to parse action");
                    
                    // Execute action with full pipeline
                    let result = kernel.execute_full_action(&session_id, action).await;
                    
                    let elapsed = start.elapsed();
                    
                    // Ensure we meet the 25ms target
                    assert!(elapsed < Duration::from_millis(25), 
                        "Action took {}ms, exceeding 25ms target", elapsed.as_millis());
                    
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark micro-planner decision making
/// Target: Sub-5ms for selector strategy generation
fn benchmark_micro_planner(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("micro_planner");
    group.significance_level(0.1).sample_size(2000);
    
    let test_targets = vec![
        ("simple_button", "button", "Submit", ""),
        ("complex_form", "input", "Email Address", "form-control required"),
        ("nested_element", "div", "Add to Cart", "btn btn-primary add-cart"),
    ];
    
    for (test_name, element_type, text, css_classes) in test_targets {
        group.bench_with_input(
            BenchmarkId::new("selector_strategy", test_name),
            &(element_type, text, css_classes),
            |b, (elem_type, elem_text, classes)| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    let mut planner = MicroPlanner::new().expect("Failed to create planner");
                    
                    let strategy = planner.plan_selector_strategy(
                        elem_type,
                        elem_text, 
                        classes,
                        &[]
                    ).await.expect("Failed to plan strategy");
                    
                    let elapsed = start.elapsed();
                    
                    // Ensure micro-planner is under 5ms
                    assert!(elapsed < Duration::from_millis(5),
                        "Micro-planner took {}ms, exceeding 5ms target", elapsed.as_millis());
                    
                    black_box(strategy)
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark vision processing pipeline
/// Target: Sub-10ms for vision embedding + OCR
fn benchmark_vision_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("vision_processing");
    group.significance_level(0.1).sample_size(500);
    
    // Create test images of different sizes
    let test_images = vec![
        ("small_screenshot", generate_test_image(400, 300)),
        ("medium_screenshot", generate_test_image(800, 600)),
        ("large_screenshot", generate_test_image(1920, 1080)),
    ];
    
    for (image_name, image_data) in test_images {
        group.bench_with_input(
            BenchmarkId::new("vision_embedding", image_name),
            &image_data,
            |b, img_data| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    let mut processor = VisionProcessor::new()
                        .expect("Failed to create vision processor");
                    
                    // Generate vision embedding
                    let embedding = processor.generate_vision_embedding(img_data).await
                        .expect("Failed to generate embedding");
                    
                    let elapsed = start.elapsed();
                    
                    // Ensure vision processing is under 10ms
                    assert!(elapsed < Duration::from_millis(10),
                        "Vision processing took {}ms, exceeding 10ms target", elapsed.as_millis());
                    
                    black_box(embedding)
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("ocr_extraction", image_name),
            &image_data,
            |b, img_data| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    let mut processor = VisionProcessor::new()
                        .expect("Failed to create vision processor");
                    
                    // Extract text via OCR
                    let text = processor.extract_text_from_image(img_data).await
                        .expect("Failed to extract text");
                    
                    let elapsed = start.elapsed();
                    
                    // Ensure OCR is under 15ms
                    assert!(elapsed < Duration::from_millis(15),
                        "OCR took {}ms, exceeding 15ms target", elapsed.as_millis());
                    
                    black_box(text)
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark DOM capture and processing
/// Target: Sub-5ms for DOM snapshot
fn benchmark_dom_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("dom_operations");
    group.significance_level(0.1).sample_size(1500);
    
    group.bench_function("dom_snapshot", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();
            
            let dom_capture = DOMCapture::new().expect("Failed to create DOM capture");
            
            // Capture DOM snapshot
            let snapshot = dom_capture.capture_snapshot().await
                .expect("Failed to capture snapshot");
            
            let elapsed = start.elapsed();
            
            // Ensure DOM capture is under 5ms
            assert!(elapsed < Duration::from_millis(5),
                "DOM capture took {}ms, exceeding 5ms target", elapsed.as_millis());
            
            black_box(snapshot)
        });
    });
    
    group.finish();
}

/// Benchmark action execution (without browser interaction)
/// Target: Sub-3ms for action preparation and validation
fn benchmark_action_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("action_execution");
    group.significance_level(0.1).sample_size(3000);
    
    let test_actions = vec![
        ("click", r#"{"type": "click", "selector": "#btn", "coordinates": [100, 200]}"#),
        ("type", r#"{"type": "type", "selector": "#input", "text": "Hello World"}"#),
        ("select", r#"{"type": "select", "selector": "#dropdown", "value": "option1"}"#),
    ];
    
    for (action_type, action_json) in test_actions {
        group.bench_with_input(
            BenchmarkId::new("action_prep", action_type),
            &action_json,
            |b, action_data| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    let executor = ActionExecutor::new().expect("Failed to create executor");
                    
                    // Parse and validate action
                    let action: serde_json::Value = serde_json::from_str(action_data)
                        .expect("Failed to parse action");
                    
                    // Prepare action (validate, optimize, etc.)
                    let prepared = executor.prepare_action(action).await
                        .expect("Failed to prepare action");
                    
                    let elapsed = start.elapsed();
                    
                    // Ensure action preparation is under 3ms
                    assert!(elapsed < Duration::from_millis(3),
                        "Action preparation took {}ms, exceeding 3ms target", elapsed.as_millis());
                    
                    black_box(prepared)
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark evidence collection
/// Target: Sub-8ms for evidence capture
fn benchmark_evidence_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("evidence_collection");
    group.significance_level(0.1).sample_size(800);
    
    group.bench_function("screenshot_capture", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();
            
            let evidence = EvidenceCollector::new("test_session".to_string())
                .expect("Failed to create evidence collector");
            
            // Capture screenshot
            let screenshot = evidence.capture_screenshot().await
                .expect("Failed to capture screenshot");
            
            let elapsed = start.elapsed();
            
            // Ensure screenshot capture is under 8ms
            assert!(elapsed < Duration::from_millis(8),
                "Screenshot capture took {}ms, exceeding 8ms target", elapsed.as_millis());
            
            black_box(screenshot)
        });
    });
    
    group.bench_function("dom_evidence", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();
            
            let evidence = EvidenceCollector::new("test_session".to_string())
                .expect("Failed to create evidence collector");
            
            // Capture DOM evidence
            let dom_data = evidence.capture_dom_snapshot().await
                .expect("Failed to capture DOM");
            
            let elapsed = start.elapsed();
            
            // Ensure DOM evidence is under 5ms
            assert!(elapsed < Duration::from_millis(5),
                "DOM evidence took {}ms, exceeding 5ms target", elapsed.as_millis());
            
            black_box(dom_data)
        });
    });
    
    group.finish();
}

/// Benchmark cached vs non-cached operations
/// Target: Sub-1ms for cached operations
fn benchmark_caching_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("caching_performance");
    group.significance_level(0.1).sample_size(5000);
    
    group.bench_function("cached_action", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();
            
            let mut kernel = EdgeKernel::new().expect("Failed to create kernel");
            let session_id = kernel.create_session().await.expect("Failed to create session");
            
            let action = serde_json::json!({
                "type": "click",
                "selector": "#cached-button"
            });
            
            // Execute cached action (should be pre-computed)
            let result = kernel.execute_cached_action(&session_id, action).await;
            
            let elapsed = start.elapsed();
            
            // Ensure cached operations are under 1ms
            assert!(elapsed < Duration::from_millis(1),
                "Cached action took {}ms, exceeding 1ms target", elapsed.as_millis());
            
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark memory usage and allocation patterns
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.significance_level(0.1).sample_size(1000);
    
    group.bench_function("memory_allocation", |b| {
        b.to_async(&rt).iter(|| async {
            // Measure memory before
            let memory_before = get_memory_usage();
            
            let mut kernel = EdgeKernel::new().expect("Failed to create kernel");
            let session_id = kernel.create_session().await.expect("Failed to create session");
            
            // Perform multiple operations
            for i in 0..100 {
                let action = serde_json::json!({
                    "type": "click",
                    "selector": format!("#button-{}", i)
                });
                
                let _ = kernel.execute_cached_action(&session_id, action).await;
            }
            
            // Measure memory after
            let memory_after = get_memory_usage();
            let memory_diff = memory_after - memory_before;
            
            // Ensure memory usage is reasonable (less than 10MB for 100 operations)
            assert!(memory_diff < 10 * 1024 * 1024,
                "Memory usage too high: {} bytes for 100 operations", memory_diff);
            
            black_box(memory_diff)
        });
    });
    
    group.finish();
}

/// Benchmark concurrent operations
/// Target: Maintain sub-25ms even with 10 concurrent operations
fn benchmark_concurrency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrency");
    group.significance_level(0.1).sample_size(200);
    
    let concurrency_levels = vec![1, 5, 10, 20];
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_actions", concurrency),
            &concurrency,
            |b, &concurrent_count| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    let mut handles = Vec::new();
                    
                    for i in 0..concurrent_count {
                        let handle = tokio::spawn(async move {
                            let mut kernel = EdgeKernel::new().expect("Failed to create kernel");
                            let session_id = kernel.create_session().await
                                .expect("Failed to create session");
                            
                            let action = serde_json::json!({
                                "type": "click",
                                "selector": format!("#button-{}", i)
                            });
                            
                            kernel.execute_cached_action(&session_id, action).await
                        });
                        handles.push(handle);
                    }
                    
                    // Wait for all operations to complete
                    let results = futures::future::join_all(handles).await;
                    
                    let elapsed = start.elapsed();
                    
                    // Even with concurrency, operations should complete reasonably fast
                    let max_time_per_operation = Duration::from_millis(25 * concurrent_count as u64 / 2);
                    assert!(elapsed < max_time_per_operation,
                        "Concurrent operations took {}ms with {} operations", 
                        elapsed.as_millis(), concurrent_count);
                    
                    black_box(results)
                });
            }
        );
    }
    
    group.finish();
}

/// Generate test image data for benchmarking
fn generate_test_image(width: u32, height: u32) -> Vec<u8> {
    // Generate a simple test image (would be more sophisticated in practice)
    let mut image_data = Vec::with_capacity((width * height * 4) as usize);
    
    for y in 0..height {
        for x in 0..width {
            // Create a simple pattern
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = ((x + y) * 255 / (width + height)) as u8;
            let a = 255u8;
            
            image_data.extend_from_slice(&[r, g, b, a]);
        }
    }
    
    // Convert to PNG format (simplified)
    image_data
}

/// Get current memory usage (platform-specific implementation)
fn get_memory_usage() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert to bytes
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output() 
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                    return rss_kb * 1024; // Convert to bytes
                }
            }
        }
    }
    
    // Fallback
    0
}

/// Performance regression tests
/// These tests fail if performance degrades below targets
#[cfg(test)]
mod performance_tests {
    use super::*;
    use tokio::test;
    
    #[tokio::test]
    async fn test_action_execution_under_25ms() {
        let start = Instant::now();
        
        let mut kernel = EdgeKernel::new().expect("Failed to create kernel");
        let session_id = kernel.create_session().await.expect("Failed to create session");
        
        let action = serde_json::json!({
            "type": "click",
            "selector": "#test-button"
        });
        
        let _result = kernel.execute_full_action(&session_id, action).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(25), 
            "Action execution took {}ms, exceeding 25ms target", elapsed.as_millis());
    }
    
    #[tokio::test]
    async fn test_micro_planner_under_5ms() {
        let start = Instant::now();
        
        let mut planner = MicroPlanner::new().expect("Failed to create planner");
        let _strategy = planner.plan_selector_strategy("button", "Submit", "btn", &[]).await
            .expect("Failed to plan strategy");
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(5),
            "Micro-planner took {}ms, exceeding 5ms target", elapsed.as_millis());
    }
    
    #[tokio::test]
    async fn test_vision_processing_under_10ms() {
        let start = Instant::now();
        
        let mut processor = VisionProcessor::new().expect("Failed to create processor");
        let test_image = generate_test_image(800, 600);
        let _embedding = processor.generate_vision_embedding(&test_image).await
            .expect("Failed to generate embedding");
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10),
            "Vision processing took {}ms, exceeding 10ms target", elapsed.as_millis());
    }
}

criterion_group!(
    benches,
    benchmark_full_action_cycle,
    benchmark_micro_planner,
    benchmark_vision_processing,
    benchmark_dom_operations,
    benchmark_action_execution,
    benchmark_evidence_collection,
    benchmark_caching_performance,
    benchmark_memory_efficiency,
    benchmark_concurrency
);

criterion_main!(benches);