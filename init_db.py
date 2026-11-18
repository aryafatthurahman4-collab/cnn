import os
import sqlite3

DB_PATH = "sensor_detection.db"
EVIDENCE_DIR = "captured_evidence"

def init_database():
    """
    Menginisialisasi tabel SQLite sensor_detection.db
    Mendukung 4 kelas: Kucing, Kelinci, Burung, Ayam
    Serta integrasi sensor kamera, AI models, Ultralytics, dan Roboflow Workflows.
    """
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Tabel Master Sensor Kamera
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_cameras (
            camera_id TEXT PRIMARY KEY,
            location_name TEXT NOT NULL,
            resolution TEXT DEFAULT '1920x1080',
            fps INTEGER DEFAULT 30,
            status TEXT DEFAULT 'ACTIVE',
            stream_url TEXT DEFAULT 'rtsp://192.168.1.101/live',
            model_engine TEXT DEFAULT 'MobileNetV3 + K-Means L*a*b*',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Tabel Master Kelas Hewan (4 Kelas: Kucing, Kelinci, Burung, Ayam)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS animal_classes (
            class_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE NOT NULL,
            scientific_name TEXT,
            icon TEXT DEFAULT 'paw',
            risk_level TEXT DEFAULT 'LOW',
            description TEXT
        )
    ''')

    # 3. Tabel AI Models (MobileNet, ResNet, YOLOv8, Roboflow)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_models (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            framework TEXT NOT NULL,
            backbone TEXT NOT NULL,
            input_shape TEXT DEFAULT '224x224x3',
            accuracy REAL DEFAULT 0.0,
            loss REAL DEFAULT 0.0,
            map50 REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            weights_path TEXT,
            status TEXT DEFAULT 'READY'
        )
    ''')

    # 4. Tabel Log Deteksi Real-Time Telemetri
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            detected_class TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            bbox_x INTEGER DEFAULT 0,
            bbox_y INTEGER DEFAULT 0,
            bbox_w INTEGER DEFAULT 0,
            bbox_h INTEGER DEFAULT 0,
            image_evidence_path TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            segmentation_method TEXT DEFAULT 'K-Means L*a*b*',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES sensor_cameras(camera_id)
        )
    ''')

    # 5. Tabel Roboflow Workflows
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS roboflow_workflows (
            workflow_id TEXT PRIMARY KEY,
            workspace TEXT NOT NULL,
            workflow_url TEXT NOT NULL,
            project_name TEXT NOT NULL,
            endpoint_version TEXT DEFAULT 'v2',
            classes_supported TEXT NOT NULL,
            status TEXT DEFAULT 'CONNECTED',
            last_sync DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 6. Tabel Ultralytics Explorer Configs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ultralytics_configs (
            config_id TEXT PRIMARY KEY,
            github_repo TEXT DEFAULT 'https://github.com/ultralytics/ultralytics',
            explore_url TEXT DEFAULT 'https://platform.ultralytics.com/explore',
            model_variant TEXT DEFAULT 'YOLOv8n',
            task TEXT DEFAULT 'detect',
            conf_threshold REAL DEFAULT 0.50,
            iou_threshold REAL DEFAULT 0.45,
            status TEXT DEFAULT 'ACTIVE'
        )
    ''')

    # SEED DATA: Sensor Cameras
    cameras = [
        ('CAM_SENSOR_01', 'Kandang Observasi A (Outdoor)', '1920x1080', 30, 'ACTIVE', 'rtsp://192.168.1.101/cam1', 'MobileNetV3 + K-Means L*a*b*'),
        ('CAM_SENSOR_02', 'Zona Pakan B (Indoor)', '1280x720', 30, 'ACTIVE', 'rtsp://192.168.1.102/cam2', 'Ultralytics YOLOv8n-Animal'),
        ('CAM_SENSOR_03', 'Pagar Perimeter C (Camera Trap)', '1920x1080', 60, 'ACTIVE', 'rtsp://192.168.1.103/cam3', 'ResNet50 PyTorch Classifier'),
        ('CAM_SENSOR_04', 'Kandang Unggas & Aviary D', '1920x1080', 30, 'ACTIVE', 'rtsp://192.168.1.104/cam4', 'Roboflow Custom Workflow Engine')
    ]
    cursor.executemany('''
        INSERT OR REPLACE INTO sensor_cameras (camera_id, location_name, resolution, fps, status, stream_url, model_engine)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', cameras)

    # SEED DATA: Animal Classes (4 Kelas)
    animal_classes = [
        ('Kucing', 'Felis catus', 'cat', 'LOW', 'Mamalia karnivora domestik, bulu beragam corak, telinga lancip segitiga, mata malam reflektif.'),
        ('Kelinci', 'Oryctolagus cuniculus', 'rabbit', 'LOW', 'Mamalia herbivora bertubuh kompak, telinga panjang tegak, gerakan lincah dan cepat.'),
        ('Burung', 'Aves', 'bird', 'LOW', 'Unggas bipedal bersayap dan berbulu, paruh khas, memiliki kecepatan terbang dan manuver tinggi.'),
        ('Ayam', 'Gallus gallus domesticus', 'rooster', 'LOW', 'Unggas domestik ternak, jengger merah (comb) di kepala, pial leher, dan paruh kokoh.')
    ]
    cursor.executemany('''
        INSERT OR REPLACE INTO animal_classes (class_name, scientific_name, icon, risk_level, description)
        VALUES (?, ?, ?, ?, ?)
    ''', animal_classes)

    # SEED DATA: AI Models
    models = [
        ('MDL-TF-01', 'Animal-MobileNetV3-KMeans', 'TensorFlow / Keras', 'MobileNetV3-Large + K-Means L*a*b*', '224x224x3', 96.85, 0.084, 94.20, 18.5, 'model_animal_sensor.h5', 'DEPLOYED'),
        ('MDL-TORCH-02', 'Animal-ResNet50-Torch', 'PyTorch 2.14', 'ResNet-50 Transfer Learning', '224x224x3', 97.40, 0.071, 95.80, 22.1, 'model_animal_sensor.pth', 'READY'),
        ('MDL-YOLO-03', 'Ultralytics YOLOv8n-Animal', 'Ultralytics YOLOv8', 'YOLOv8 Nano (Anchor-Free)', '640x640x3', 95.60, 0.112, 93.90, 14.2, 'runs/detect/yolov8n_animal/weights/best.pt', 'DEPLOYED'),
        ('MDL-ROBO-04', 'Roboflow Multi-Animal Workflow', 'Roboflow Workflows', 'Roboflow Hosted Detection API', '640x640x3', 98.10, 0.052, 97.40, 28.0, 'https://app.roboflow.com/muhammad-arya-fatthurahman/workflows', 'ACTIVE')
    ]
    cursor.executemany('''
        INSERT OR REPLACE INTO ai_models (model_id, model_name, framework, backbone, input_shape, accuracy, loss, map50, latency_ms, weights_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', models)

    # SEED DATA: Roboflow Workflows
    cursor.execute('''
        INSERT OR REPLACE INTO roboflow_workflows (workflow_id, workspace, workflow_url, project_name, endpoint_version, classes_supported, status)
        VALUES ('wf-animal-detect-arya', 'muhammad-arya-fatthurahman', 'https://app.roboflow.com/muhammad-arya-fatthurahman/workflows', 'animal-detector-cnn-yolo', 'v2', 'Kucing, Burung, Kelinci, Ayam', 'CONNECTED')
    ''')

    # SEED DATA: Ultralytics Configs
    cursor.execute('''
        INSERT OR REPLACE INTO ultralytics_configs (config_id, github_repo, explore_url, model_variant, task, conf_threshold, iou_threshold, status)
        VALUES ('CFG-YOLO-01', 'https://github.com/ultralytics/ultralytics', 'https://platform.ultralytics.com/explore', 'YOLOv8n', 'detect', 0.50, 0.45, 'ACTIVE')
    ''')

    # SEED DATA: Detection Logs (Sample Real Telemetry untuk Semua 4 Kelas)
    sample_logs = [
        ('CAM_SENSOR_01', 'Kucing', 95.40, 110, 75, 250, 265, 'captured_evidence/Kucing_20260922_182214.jpg', 18.2, 'K-Means L*a*b*', '2026-09-22 18:22:14'),
        ('CAM_SENSOR_02', 'Kelinci', 97.20, 140, 95, 220, 240, 'captured_evidence/Kelinci_20260922_182214.jpg', 19.5, 'K-Means L*a*b*', '2026-09-22 18:22:14'),
        ('CAM_SENSOR_03', 'Burung', 91.80, 190, 45, 175, 185, 'captured_evidence/Burung_20260922_182214.jpg', 21.0, 'YOLOv8-BBox', '2026-09-22 18:22:14'),
        ('CAM_SENSOR_04', 'Ayam', 96.50, 125, 60, 230, 255, 'captured_evidence/Ayam_20260923_001000.jpg', 17.8, 'Roboflow-Workflow', '2026-09-23 00:10:00'),
        ('CAM_SENSOR_01', 'Kucing', 93.15, 115, 80, 245, 260, 'captured_evidence/Kucing_20260922_182214.jpg', 18.0, 'K-Means L*a*b*', '2026-09-23 00:11:30'),
        ('CAM_SENSOR_04', 'Ayam', 98.20, 130, 55, 235, 260, 'captured_evidence/Ayam_20260923_001000.jpg', 16.9, 'Ultralytics YOLOv8', '2026-09-23 00:12:15')
    ]
    cursor.executemany('''
        INSERT INTO detection_logs 
        (camera_id, detected_class, confidence_score, bbox_x, bbox_y, bbox_w, bbox_h, image_evidence_path, latency_ms, segmentation_method, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_logs)

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Database SQLite '{DB_PATH}' berhasil diinisialisasi untuk 4 kelas hewan + Roboflow + Ultralytics!")

if __name__ == "__main__":
    init_database()
