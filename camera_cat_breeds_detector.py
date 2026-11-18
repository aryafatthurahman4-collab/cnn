"""
Sistem Kamera Deteksi 12 Ras Kucing dengan ResNet50V2
Referensi: JIEET Volume 08 Nomor 01, 2024
Ras: Abyssinia, Bengal, Bombay, Birman, British Shorthair, Egyptian, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx
"""

import os
import cv2
import time
import sqlite3
import argparse
import numpy as np
import tensorflow as tf

# ==========================================================
# 1. KONFIGURASI SISTEM & DATABASE
# ==========================================================
CAMERA_ID = "CAT_BREEDS_CAM_01"
CAMERA_LOCATION = "Observasi Ras Kucing"
DB_PATH = "cat_breeds_detection.db"
EVIDENCE_DIR = "cat_breeds_evidence"
CONFIDENCE_THRESHOLD = 0.80  # Ambang batas confidence
CAPTURE_INTERVAL_SEC = 3.0    # Jeda cooldown pencatatan
IMG_SIZE = (224, 224)

# 12 Ras Kucing sesuai JIEET 2024
CAT_BREEDS = [
    "Abyssinia",
    "Bengal", 
    "Bombay",
    "Birman",
    "British Shorthair",
    "Egyptian",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx"
]

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Inisialisasi Database SQLite
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cat_breed_detections (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            detected_breed TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            bbox_x INTEGER DEFAULT 0,
            bbox_y INTEGER DEFAULT 0,
            bbox_w INTEGER DEFAULT 0,
            bbox_h INTEGER DEFAULT 0,
            image_evidence_path TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# ==========================================================
# 2. MEMUAT MODEL RESNET50V2 PRE-TRAINED
# ==========================================================
print("[INFO] Memuat model ResNet50V2 untuk klasifikasi 12 ras kucing...")

model = None
try:
    # Coba load model yang sudah dilatih
    if os.path.exists("cat_breeds_resnet50v2_final.h5"):
        model = tf.keras.models.load_model("cat_breeds_resnet50v2_final.h5")
        print("[INFO] Model 'cat_breeds_resnet50v2_final.h5' berhasil dimuat.")
    elif os.path.exists("best_cat_breeds_resnet50v2.h5"):
        model = tf.keras.models.load_model("best_cat_breeds_resnet50v2.h5")
        print("[INFO] Model 'best_cat_breeds_resnet50v2.h5' berhasil dimuat.")
    else:
        print("[WARNING] Model training belum ditemukan. Menggunakan ResNet50V2 ImageNet...")
        # Fallback ke ResNet50V2 ImageNet jika model training belum ada
        base_model = tf.keras.applications.ResNet50V2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(len(CAT_BREEDS), activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        print("[INFO] Menggunakan model ResNet50V2 ImageNet sebagai fallback.")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    exit(1)

# ==========================================================
# 3. LOOPING SENSOR DETEKSI KAMERA REAL-TIME
# ==========================================================
def run_cat_breeds_camera_system(video_source=0, max_frames=None, headless=False):
    """
    Menjalankan loop sensor kamera untuk deteksi 12 ras kucing
    video_source: 0 (webcam), path video mp4, atau 'test' (simulasi)
    """
    print(f"\n[INFO] Mengaktifkan Sensor Kamera: {CAMERA_ID} di {CAMERA_LOCATION}")
    print(f"[INFO] Target: Deteksi 12 Ras Kucing dengan ResNet50V2")
    
    if isinstance(video_source, str) and not os.path.exists(video_source) and video_source != "test":
        print(f"[WARNING] File video '{video_source}' tidak ditemukan, beralih ke simulasi...")
        video_source = "test"

    use_test_stream = (video_source == "test")
    if not use_test_stream:
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("[WARNING] Kamera fisik tidak terdeteksi. Beralih ke mode test...")
            use_test_stream = True

    last_logged_time = 0
    frame_count = 0

    # Daftar citra test jika kamera fisik tidak tersedia
    test_images = []
    if use_test_stream:
        for breed in CAT_BREEDS:
            folder = os.path.join("dataset_cat_breeds", breed)
            if os.path.exists(folder):
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
                test_images.extend(files)
        if not test_images:
            print("[ERROR] Dataset test tidak ditemukan!")
            return

    print("[INFO] Sensor Kamera Aktif! Tekan 'q' pada jendela visualisasi untuk berhenti.")

    try:
        while True:
            start_time = time.time()

            if use_test_stream:
                img_path = test_images[frame_count % len(test_images)]
                frame = cv2.imread(img_path)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (640, 480))
                time.sleep(0.04)  # Simulasi ~25 FPS
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Stream kamera selesai atau terputus.")
                    break

            frame_count += 1
            display_frame = frame.copy()

            # Preprocessing Input ResNet50V2
            resized = cv2.resize(frame, IMG_SIZE)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm_frame = np.expand_dims(rgb_frame / 255.0, axis=0)

            # Inferensi ResNet50V2
            predictions = model.predict(norm_frame, verbose=0)[0]
            class_idx = int(np.argmax(predictions))
            detected_breed = CAT_BREEDS[class_idx]
            confidence = float(predictions[class_idx])

            # Estimasi bounding box (center region)
            h_frame, w_frame = frame.shape[:2]
            bbox_x = int(w_frame * 0.2)
            bbox_y = int(h_frame * 0.2)
            bbox_w = int(w_frame * 0.6)
            bbox_h = int(h_frame * 0.6)

            latency_ms = (time.time() - start_time) * 1000
            current_fps = 1000.0 / (latency_ms if latency_ms > 0 else 1)

            # Simpan ke Database jika confidence >= threshold
            current_time = time.time()
            if confidence >= CONFIDENCE_THRESHOLD and (current_time - last_logged_time) > CAPTURE_INTERVAL_SEC:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                breed_filename = detected_breed.replace(" ", "_")
                img_filename = f"{breed_filename}_{timestamp_str}.jpg"
                img_full_path = os.path.join(EVIDENCE_DIR, img_filename)
                cv2.imwrite(img_full_path, frame)

                # Simpan record ke database
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO cat_breed_detections 
                    (camera_id, detected_breed, confidence_score, bbox_x, bbox_y, bbox_w, bbox_h, image_evidence_path, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    CAMERA_ID,
                    detected_breed,
                    round(confidence * 100, 2),
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    img_full_path,
                    round(latency_ms, 2)
                ))
                conn.commit()
                conn.close()

                last_logged_time = current_time
                print(f"[DATABASE LOG] Deteksi: {detected_breed} ({confidence*100:.2f}%) | Latensi: {latency_ms:.1f}ms")

            # Render Visualisasi HUD
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            
            # Gambar Bounding Box
            cv2.rectangle(display_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, 2)
            cv2.putText(display_frame, f"{detected_breed} {confidence*100:.1f}%", 
                        (bbox_x, max(25, bbox_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Header Status
            cv2.putText(display_frame, f"ResNet50V2 Cat Breeds Detector", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"12 Breeds: {detected_breed}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display_frame, f"Confidence: {confidence*100:.1f}% | Latency: {latency_ms:.1f}ms | FPS: {current_fps:.1f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Tampilkan UI OpenCV jika tidak headless
            if not headless:
                cv2.imshow("ResNet50V2 Cat Breeds Detection - Real-time", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if max_frames and frame_count >= max_frames:
                print(f"[INFO] Batas pengujian ({max_frames} frame) tercapai.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Sensor dihentikan oleh pengguna.")
    finally:
        if not use_test_stream:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("[INFO] Sistem sensor kamera dinonaktifkan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistem Deteksi 12 Ras Kucing dengan ResNet50V2")
    parser.add_argument("--source", default="0", help="Sumber video: '0' untuk webcam, path file mp4, atau 'test' untuk simulasi")
    parser.add_argument("--frames", type=int, default=None, help="Jumlah frame maksimal (opsional)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_cat_breeds_camera_system(video_source=src, max_frames=args.frames, headless=args.headless)
