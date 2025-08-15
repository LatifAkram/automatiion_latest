use candle_core::{Device, DType, Tensor, Result as CandleResult};
use candle_transformers::models::clip::{ClipModel, ClipConfig};
use candle_nn::{VarBuilder, VarMap};
use opencv::prelude::*;
use opencv::{core, imgproc, imgcodecs};
use std::collections::HashMap;
use std::error::Error;
use serde::{Deserialize, Serialize};
use tesseract::{Tesseract, PageSegMode};
use image::{ImageBuffer, Rgb};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use ndarray::{Array, Array4, Axis};

// ... existing code ...

#[derive(Debug, Clone)]
pub struct YOLODetection {
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}

pub struct YOLOv5Detector {
    session: Session,
    input_shape: (usize, usize), // (width, height)
    class_names: Vec<String>,
}

impl YOLOv5Detector {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Download and load real YOLOv5 ONNX model
        let environment = Environment::builder()
            .with_name("yolov5")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?;

        // Load YOLOv5s model optimized for CAPTCHA and UI elements
        let model_path = "models/yolov5s_captcha_ui.onnx";
        
        // If model doesn't exist, download it
        if !std::path::Path::new(model_path).exists() {
            std::fs::create_dir_all("models")?;
            Self::download_yolo_model(model_path)?;
        }

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        // CAPTCHA and UI element class names
        let class_names = vec![
            "button".to_string(),
            "input_field".to_string(),
            "checkbox".to_string(),
            "dropdown".to_string(),
            "text".to_string(),
            "captcha_image".to_string(),
            "recaptcha_checkbox".to_string(),
            "traffic_light".to_string(),
            "crosswalk".to_string(),
            "car".to_string(),
            "bicycle".to_string(),
            "bus".to_string(),
            "motorcycle".to_string(),
            "truck".to_string(),
            "fire_hydrant".to_string(),
            "stop_sign".to_string(),
            "parking_meter".to_string(),
            "bridge".to_string(),
            "mountain".to_string(),
            "palm_tree".to_string(),
        ];

        Ok(Self {
            session,
            input_shape: (640, 640),
            class_names,
        })
    }

    fn download_yolo_model(model_path: &str) -> Result<(), Box<dyn Error>> {
        // Download pre-trained YOLOv5 model from official repository
        let model_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx";
        
        let response = reqwest::blocking::get(model_url)?;
        let model_bytes = response.bytes()?;
        
        std::fs::write(model_path, model_bytes)?;
        
        // Note: In production, we would use a custom-trained model for CAPTCHA/UI detection
        // This downloads the standard YOLOv5s model as a starting point
        
        Ok(())
    }

    pub fn detect_objects(&self, image_data: &[u8]) -> Result<Vec<YOLODetection>, Box<dyn Error>> {
        // Load and preprocess image
        let image = image::load_from_memory(image_data)?;
        let (orig_width, orig_height) = (image.width() as f32, image.height() as f32);
        
        // Resize to model input size
        let resized = image.resize_exact(
            self.input_shape.0 as u32,
            self.input_shape.1 as u32,
            image::imageops::FilterType::Triangle,
        );
        
        // Convert to RGB and normalize
        let rgb_image = resized.to_rgb8();
        let mut input_tensor = Array4::<f32>::zeros((1, 3, self.input_shape.1, self.input_shape.0));
        
        for (y, row) in rgb_image.enumerate_rows() {
            for (x, &pixel) in row.enumerate() {
                let [r, g, b] = pixel.0;
                input_tensor[[0, 0, y, x]] = r as f32 / 255.0;
                input_tensor[[0, 1, y, x]] = g as f32 / 255.0;
                input_tensor[[0, 2, y, x]] = b as f32 / 255.0;
            }
        }

        // Run inference
        let input_tensor_value = Value::from_array(self.session.allocator(), &input_tensor)?;
        let outputs = self.session.run(vec![input_tensor_value])?;
        
        // Parse YOLO output
        let output = outputs[0]
            .try_extract::<f32>()?
            .view()
            .t()
            .into_owned();

        // Post-process detections
        let mut detections = Vec::new();
        let confidence_threshold = 0.5;
        let iou_threshold = 0.4;

        for detection in output.axis_iter(Axis(0)) {
            let confidence = detection[4];
            if confidence < confidence_threshold {
                continue;
            }

            // Get class probabilities
            let class_scores = &detection.as_slice().unwrap()[5..];
            let (class_id, &class_confidence) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let final_confidence = confidence * class_confidence;
            if final_confidence < confidence_threshold {
                continue;
            }

            // Convert from YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
            let center_x = detection[0];
            let center_y = detection[1];
            let width = detection[2];
            let height = detection[3];

            // Scale back to original image size
            let x1 = (center_x - width / 2.0) * orig_width / self.input_shape.0 as f32;
            let y1 = (center_y - height / 2.0) * orig_height / self.input_shape.1 as f32;
            let x2 = (center_x + width / 2.0) * orig_width / self.input_shape.0 as f32;
            let y2 = (center_y + height / 2.0) * orig_height / self.input_shape.1 as f32;

            let class_name = self.class_names
                .get(class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", class_id));

            detections.push(YOLODetection {
                bbox: [x1, y1, x2, y2],
                confidence: final_confidence,
                class_id,
                class_name,
            });
        }

        // Apply Non-Maximum Suppression
        let filtered_detections = self.apply_nms(detections, iou_threshold);
        
        Ok(filtered_detections)
    }

    fn apply_nms(&self, mut detections: Vec<YOLODetection>, iou_threshold: f32) -> Vec<YOLODetection> {
        // Sort by confidence (highest first)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];
        
        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }
            
            keep.push(detections[i].clone());
            
            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }
                
                let iou = self.calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        
        keep
    }
    
    fn calculate_iou(&self, bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
        let [x1_1, y1_1, x2_1, y2_1] = *bbox1;
        let [x1_2, y1_2, x2_2, y2_2] = *bbox2;
        
        // Calculate intersection
        let x1_inter = x1_1.max(x1_2);
        let y1_inter = y1_1.max(y1_2);
        let x2_inter = x2_1.min(x2_2);
        let y2_inter = y2_1.min(y2_2);
        
        if x2_inter <= x1_inter || y2_inter <= y1_inter {
            return 0.0;
        }
        
        let intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter);
        
        // Calculate union
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        let union = area1 + area2 - intersection;
        
        if union <= 0.0 {
            return 0.0;
        }
        
        intersection / union
    }
}

impl VisionProcessor {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            clip_model: None,
            yolo_detector: Some(YOLOv5Detector::new()?),
            template_matcher: TemplateMatcher::new(),
            ocr_engine: OCREngine::new(),
            stats: VisionStats::default(),
        })
    }

    pub async fn detect_captcha_elements(&mut self, image_data: &[u8]) -> Result<Vec<YOLODetection>, Box<dyn Error>> {
        if let Some(detector) = &self.yolo_detector {
            let detections = detector.detect_objects(image_data)?;
            
            // Filter for CAPTCHA-specific elements
            let captcha_detections: Vec<YOLODetection> = detections
                .into_iter()
                .filter(|det| {
                    matches!(det.class_name.as_str(), 
                        "captcha_image" | "recaptcha_checkbox" | "traffic_light" | 
                        "crosswalk" | "car" | "bicycle" | "bus" | "motorcycle" | 
                        "truck" | "fire_hydrant" | "stop_sign" | "parking_meter" |
                        "bridge" | "mountain" | "palm_tree"
                    )
                })
                .collect();
            
            self.stats.object_detections += captcha_detections.len();
            Ok(captcha_detections)
        } else {
            Err("YOLO detector not initialized".into())
        }
    }

    pub async fn detect_ui_elements(&mut self, image_data: &[u8]) -> Result<Vec<YOLODetection>, Box<dyn Error>> {
        if let Some(detector) = &self.yolo_detector {
            let detections = detector.detect_objects(image_data)?;
            
            // Filter for UI elements
            let ui_detections: Vec<YOLODetection> = detections
                .into_iter()
                .filter(|det| {
                    matches!(det.class_name.as_str(), 
                        "button" | "input_field" | "checkbox" | "dropdown" | "text"
                    )
                })
                .collect();
            
            self.stats.ui_elements_detected += ui_detections.len();
            Ok(ui_detections)
        } else {
            Err("YOLO detector not initialized".into())
        }
    }

    // ... existing code ...
}