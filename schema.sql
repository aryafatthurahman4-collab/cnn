-- ==============================================================================
-- SKEMA DATABASE RELASIONAL LENGKAP: db_sensor_hewan
-- Sistem CNN Terintegrasi Sensor Kamera, K-Means L*a*b*, YOLOv8 Ultralytics & Roboflow
-- Kelas Terdeteksi: Kucing, Kelinci, Burung, Ayam
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS db_sensor_hewan;
USE db_sensor_hewan;

-- 1. TABEL MASTER SENSOR KAMERA
CREATE TABLE IF NOT EXISTS sensor_cameras (
    camera_id VARCHAR(50) PRIMARY KEY,
    location_name VARCHAR(100) NOT NULL,
    resolution VARCHAR(20) DEFAULT '1920x1080',
    fps INT DEFAULT 30,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    stream_url VARCHAR(255) DEFAULT 'rtsp://192.168.1.101/live',
    model_engine VARCHAR(100) DEFAULT 'MobileNetV3 + K-Means L*a*b*',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO sensor_cameras (camera_id, location_name, resolution, fps, status, stream_url, model_engine) VALUES
('CAM_SENSOR_01', 'Kandang Observasi A (Outdoor)', '1920x1080', 30, 'ACTIVE', 'rtsp://192.168.1.101/cam1', 'MobileNetV3 + K-Means L*a*b*'),
('CAM_SENSOR_02', 'Zona Pakan B (Indoor)', '1280x720', 30, 'ACTIVE', 'rtsp://192.168.1.102/cam2', 'Ultralytics YOLOv8n-Animal'),
('CAM_SENSOR_03', 'Pagar Perimeter C (Camera Trap)', '1920x1080', 60, 'ACTIVE', 'rtsp://192.168.1.103/cam3', 'ResNet50 PyTorch Classifier'),
('CAM_SENSOR_04', 'Kandang Unggas & Aviary D', '1920x1080', 30, 'ACTIVE', 'rtsp://192.168.1.104/cam4', 'Roboflow Custom Workflow Engine')
ON DUPLICATE KEY UPDATE location_name=VALUES(location_name), model_engine=VALUES(model_engine);

-- 2. TABEL MASTER KELAS HEWAN (4 KELAS)
CREATE TABLE IF NOT EXISTS animal_classes (
    class_id INT AUTO_INCREMENT PRIMARY KEY,
    class_name VARCHAR(50) UNIQUE NOT NULL,
    scientific_name VARCHAR(100),
    icon VARCHAR(50) DEFAULT 'paw',
    risk_level VARCHAR(20) DEFAULT 'LOW',
    description TEXT
);

INSERT INTO animal_classes (class_name, scientific_name, icon, risk_level, description) VALUES
('Kucing', 'Felis catus', 'cat', 'LOW', 'Mamalia karnivora domestik, bulu beragam corak, telinga segitiga tegak, mata tajam.'),
('Kelinci', 'Oryctolagus cuniculus', 'rabbit', 'LOW', 'Mamalia herbivora bertubuh kompak, telinga panjang tegak, gerakan lincah di semak.'),
('Burung', 'Aves', 'bird', 'LOW', 'Unggas bipedal berbulu, memiliki sayap dan paruh khas, frekuensi gerakan tinggi.'),
('Ayam', 'Gallus gallus domesticus', 'rooster', 'LOW', 'Unggas domestik ternak, memiliki jengger merah (comb) di kepala, pial leher, dan paruh kokoh.')
ON DUPLICATE KEY UPDATE scientific_name=VALUES(scientific_name), description=VALUES(description);

-- 3. TABEL MODEL ARSITEKTUR AI
CREATE TABLE IF NOT EXISTS ai_models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    backbone VARCHAR(100) NOT NULL,
    input_shape VARCHAR(50) DEFAULT '224x224x3',
    accuracy FLOAT DEFAULT 0.0,
    loss FLOAT DEFAULT 0.0,
    map50 FLOAT DEFAULT 0.0,
    latency_ms FLOAT DEFAULT 0.0,
    weights_path VARCHAR(255),
    status VARCHAR(20) DEFAULT 'READY'
);

INSERT INTO ai_models (model_id, model_name, framework, backbone, input_shape, accuracy, loss, map50, latency_ms, weights_path, status) VALUES
('MDL-TF-01', 'Animal-MobileNetV3-KMeans', 'TensorFlow / Keras', 'MobileNetV3-Large + K-Means L*a*b*', '224x224x3', 96.85, 0.084, 94.20, 18.5, 'model_animal_sensor.h5', 'DEPLOYED'),
('MDL-TORCH-02', 'Animal-ResNet50-Torch', 'PyTorch 2.14', 'ResNet-50 Transfer Learning', '224x224x3', 97.40, 0.071, 95.80, 22.1, 'model_animal_sensor.pth', 'READY'),
('MDL-YOLO-03', 'Ultralytics YOLOv8n-Animal', 'Ultralytics YOLOv8', 'YOLOv8 Nano (Anchor-Free)', '640x640x3', 95.60, 0.112, 93.90, 14.2, 'runs/detect/yolov8n_animal/weights/best.pt', 'DEPLOYED'),
('MDL-ROBO-04', 'Roboflow Multi-Animal Workflow', 'Roboflow Workflows', 'Roboflow Hosted Detection API', '640x640x3', 98.10, 0.052, 97.40, 28.0, 'https://app.roboflow.com/muhammad-arya-fatthurahman/workflows', 'ACTIVE')
ON DUPLICATE KEY UPDATE accuracy=VALUES(accuracy), status=VALUES(status);

-- 4. TABEL LOG DETEKSI REAL-TIME TELEMETRI
CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    camera_id VARCHAR(50) NOT NULL,
    detected_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
    bbox_x INT DEFAULT 0,
    bbox_y INT DEFAULT 0,
    bbox_w INT DEFAULT 0,
    bbox_h INT DEFAULT 0,
    image_evidence_path VARCHAR(255) NOT NULL,
    latency_ms FLOAT NOT NULL,
    segmentation_method VARCHAR(50) DEFAULT 'K-Means L*a*b*',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES sensor_cameras(camera_id) ON DELETE CASCADE
);

INSERT INTO detection_logs (camera_id, detected_class, confidence_score, bbox_x, bbox_y, bbox_w, bbox_h, image_evidence_path, latency_ms, segmentation_method, timestamp) VALUES
('CAM_SENSOR_01', 'Kucing', 95.40, 110, 75, 250, 265, 'captured_evidence/Kucing_20260922_182214.jpg', 18.2, 'K-Means L*a*b*', '2026-09-22 18:22:14'),
('CAM_SENSOR_02', 'Kelinci', 97.20, 140, 95, 220, 240, 'captured_evidence/Kelinci_20260922_182214.jpg', 19.5, 'K-Means L*a*b*', '2026-09-22 18:22:14'),
('CAM_SENSOR_03', 'Burung', 91.80, 190, 45, 175, 185, 'captured_evidence/Burung_20260922_182214.jpg', 21.0, 'YOLOv8-BBox', '2026-09-22 18:22:14'),
('CAM_SENSOR_04', 'Ayam', 96.50, 125, 60, 230, 255, 'captured_evidence/Ayam_20260923_001000.jpg', 17.8, 'Roboflow-Workflow', '2026-09-23 00:10:00'),
('CAM_SENSOR_01', 'Kucing', 93.15, 115, 80, 245, 260, 'captured_evidence/Kucing_20260922_182214.jpg', 18.0, 'K-Means L*a*b*', '2026-09-23 00:11:30'),
('CAM_SENSOR_04', 'Ayam', 98.20, 130, 55, 235, 260, 'captured_evidence/Ayam_20260923_001000.jpg', 16.9, 'Ultralytics YOLOv8', '2026-09-23 00:12:15');

-- 5. TABEL INTEGRASI ROBOFLOW WORKFLOWS
CREATE TABLE IF NOT EXISTS roboflow_workflows (
    workflow_id VARCHAR(100) PRIMARY KEY,
    workspace VARCHAR(100) NOT NULL,
    workflow_url VARCHAR(255) NOT NULL,
    project_name VARCHAR(100) NOT NULL,
    endpoint_version VARCHAR(20) DEFAULT 'v2',
    classes_supported TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'CONNECTED',
    last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO roboflow_workflows (workflow_id, workspace, workflow_url, project_name, endpoint_version, classes_supported, status) VALUES
('wf-animal-detect-arya', 'muhammad-arya-fatthurahman', 'https://app.roboflow.com/muhammad-arya-fatthurahman/workflows', 'animal-detector-cnn-yolo', 'v2', 'Kucing, Burung, Kelinci, Ayam', 'CONNECTED')
ON DUPLICATE KEY UPDATE status=VALUES(status);

-- 6. TABEL ULTRALYTICS EXPLORE & CONFIG
CREATE TABLE IF NOT EXISTS ultralytics_configs (
    config_id VARCHAR(50) PRIMARY KEY,
    github_repo VARCHAR(255) DEFAULT 'https://github.com/ultralytics/ultralytics',
    explore_url VARCHAR(255) DEFAULT 'https://platform.ultralytics.com/explore',
    model_variant VARCHAR(50) DEFAULT 'YOLOv8n',
    task VARCHAR(50) DEFAULT 'detect',
    conf_threshold FLOAT DEFAULT 0.50,
    iou_threshold FLOAT DEFAULT 0.45,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

INSERT INTO ultralytics_configs (config_id, github_repo, explore_url, model_variant, task, conf_threshold, iou_threshold, status) VALUES
('CFG-YOLO-01', 'https://github.com/ultralytics/ultralytics', 'https://platform.ultralytics.com/explore', 'YOLOv8n', 'detect', 0.50, 0.45, 'ACTIVE')
ON DUPLICATE KEY UPDATE conf_threshold=VALUES(conf_threshold);
