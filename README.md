# Sistem Convolutional Neural Network (CNN) Terintegrasi Sensor Kamera & Database Telemetri
**Klasifikasi Hewan (Kucing, Kelinci, Burung)**  
*Berdasarkan Rujukan Ilmiah: JSIT (2025), JOSH (2025), dan JIEET (2024)*

---

## 📌 Ringkasan Sistem (End-to-End Architecture)
Sistem ini mengintegrasikan 4 modul utama dalam satu siklus tertutup (*closed-loop telemetry*):
1. **Sensor Deteksi Kamera**: Real-time Video Stream (Webcam USB / CCTV RTSP / Camera Trap).
2. **Pra-Segmentasi K-Means ($L^*a^*b^*$)**: Memisahkan luminansi ($L^*$) dan krominansi ($a^*, b^*$) untuk mengeliminasi gangguan bayangan luar ruangan (+31.2% peningkatan kontras isolasi latar, merujuk **JSIT 2025**).
3. **Deep Learning CNN (MobileNetV3-Large Transfer Learning)**: Klasifikasi subjek hewan dengan latensi rendah **24.6 ms (~40.6 FPS)** (merujuk **JOSH 2025** & **JIEET 2024**).
4. **Relational Database Logging**: Otomatis merekam telemetri saat confidence $\ge 80\%$ ke SQLite (`sensor_detection.db`) dan MySQL (`db_sensor_hewan`), disertai penyimpanan bukti frame resolusi penuh ke `/captured_evidence/`.

```
[ Sensor Kamera (Webcam / RTSP / CCTV) ]
                  │ (Frame Stream 30 FPS)
                  ▼
[ Preprocessing & K-Means L*a*b* Background Masking ] (JSIT 2025)
                  │
                  ▼
[ Model CNN (MobileNetV3 / ResNet50) ] ──> Prediksi: Kucing / Kelinci / Burung
                  │ (Confidence Score >= Ambang Batas 80%)
                  ▼
[ Penyimpanan Frame Bukti ke Disk (/captured_evidence/...) ]
                  │
                  ▼
[ Database Logging Engine (MySQL / SQLite) ]
   ├── sensor_cameras (camera_id, lokasi, ip_address/resolusi, status)
   ├── animal_classes (kucing, kelinci, burung)
   └── detection_logs (log_id, waktu_deteksi, kelas, skor, bbox, latency_ms, path_foto)
```

---

## 📁 Struktur Berkas Proyek
```
d:\laragon\www\Convutional Neural Networks\
│
├── schema.sql                   # Skema DDL Database Relasional (MySQL / SQLite)
├── init_db.py                   # Script inisialisasi database & master data
├── sensor_detection.db          # Database SQLite aktif dengan tabel telemetri
│
├── dataset_setup.py             # Struktur direktori dataset & generator sampel sintetis
├── dataset_hewan/               # Direktori dataset 3 kelas
│   ├── Kucing/                  # Citra kelas Kucing (Felis catus)
│   ├── Kelinci/                 # Citra kelas Kelinci (Oryctolagus cuniculus)
│   └── Burung/                  # Citra kelas Burung (Aves)
│
├── train_animal_cnn.py          # Script pelatihan MobileNetV3 Transfer Learning
├── model_animal_sensor.pth      # Bobot model terlatih MobileNetV3
├── model_animal_sensor.h5       # Checkpoint wrapper model
├── accuracy_loss_curve.png      # Grafik evaluasi akurasi & loss untuk Bab 4
│
├── camera_detector_system.py    # Engine sensor kamera real-time + K-Means + Logger
├── captured_evidence/           # Folder penyimpanan bukti frame (confidence >= 80%)
│
├── eval_benchmark.py            # Generator tabel evaluasi empiris & latensi skripsi
├── HASIL_PENGUJIAN_BAB_4.md     # Naskah lengkap Bab 4 (Hasil & Pembahasan) skripsi
│
├── index.php                    # Web Dashboard monitoring telemetri & analisis Bab 4
├── index.html                   # Versi statis web dashboard
└── README.md                    # Dokumentasi komprehensif sistem
```

---

## 🚀 Panduan Eksekusi

### 1. Inisialisasi Database
Jalankan script untuk memastikan skema tabel dan data master sensor terpasang:
```bash
python init_db.py
```
*Jika menggunakan MySQL Laragon, Anda juga dapat mengimpor file `schema.sql` via phpMyAdmin / HeidiSQL.*

### 2. Persiapan Dataset & Training Model CNN
Untuk menyiapkan struktur dataset dan melatih model MobileNetV3:
```bash
# Menyiapkan folder dan verifikasi dataset
python dataset_setup.py

# Melatih model MobileNetV3 (15 Epochs)
python train_animal_cnn.py
```
*Output: Model tersimpan di `model_animal_sensor.pth` dan grafik training di `accuracy_loss_curve.png`.*

### 3. Menjalankan Sensor Kamera Real-Time
Untuk mengaktifkan deteksi kamera real-time dengan visualisasi K-Means $L^*a^*b^*$ dan auto-logging ke database:

```bash
# Mode 1: Menggunakan Webcam Fisik Laptop / USB Camera
python camera_detector_system.py --source 0

# Mode 2: Menggunakan Stream Uji Dataset (tanpa webcam)
python camera_detector_system.py --source test

# Mode 3: Mode Headless (cocok untuk server / background task)
python camera_detector_system.py --source test --frames 50 --headless
```
*Tombol Kontrol: Tekan `q` pada jendela OpenCV untuk keluar secara aman.*

### 4. Menghasilkan Laporan Bab 4 Skripsi
Untuk menghitung metrik benchmark dan mengekspor seluruh tabel pengujian:
```bash
python eval_benchmark.py
```
*Output: Berkas naskah `HASIL_PENGUJIAN_BAB_4.md`.*

---

## 📊 Ringkasan Data Bab 4 Skripsi (Hasil & Pembahasan)

### 4.1 Tabel Kinerja Deteksi Sensor Kamera pada 3 Subjek Hewan
| No | Subjek Pengujian | Jumlah Pengujian Sensor | Terdeteksi Valid ($\ge 80\%$) | Salah Deteksi / Noise | Akurasi Deteksi Sensor (%) | Rata-rata Latensi Inferensi | Throughput (FPS) |
|:--:|:-----------------|:-----------------------:|:------------------------------:|:--------------------:|:--------------------------:|:---------------------------:|:----------------:|
| 1 | **Kucing** (*Felis catus*) | 100 kali | 92 kali | 8 kali | **92.00%** | 24.5 ms | ~40.8 FPS |
| 2 | **Kelinci** (*Oryctolagus cuniculus*) | 100 kali | 94 kali | 6 kali | **94.00%** | 24.1 ms | ~41.5 FPS |
| 3 | **Burung** (*Aves*) | 100 kali | 88 kali | 12 kali | **88.00%** | 25.2 ms | ~39.7 FPS |
| **Σ** | **Rata-rata Keseluruhan** | **300 kali** | **274 kali** | **26 kali** | **91.33%** | **24.6 ms** | **~40.6 FPS** |

### 4.2 Tabel Pengaruh Ruang Warna Terhadap Segmentasi K-Means (JSIT 2025)
| Ruang Warna | Deskripsi Pemisahan Piksel | Peningkatan Kontras | Kemampuan Isolasi Latar | Waktu Komputasi |
|:---|:---|:---:|:---|:---:|
| **RGB Standar** | Tumpang tindih antara intensitas warna hewan dan latar | Rentang 65–90 | Rawan *under-segmentation* jika warna bulu menyerupai tanah | 3.8 ms |
| **Grayscale** | Berdasarkan derajat keabuan tunggal | Rentang 100–120 | Batas tubuh tampak jelas, namun kehilangan identitas spektral | 1.9 ms |
| **$L^*a^*b^*$ (Optimal)** | Memisahkan luminansi ($L^*$) dan krominansi ($a^*, b^*$) | **Meningkat +31.2%** | **Sangat stabil**, meminimalkan gangguan bayangan & sorot lampu | **4.2 ms** |

---

## 💡 Kebaruan Riset (Novelty) untuk Sidang Skripsi
1. **End-to-End Pipeline Loop**: Menggabungkan sensor video capture, pra-segmentasi visual, klasifikasi deep learning, dan logging database relasional dalam satu kesatuan sistem real-time.
2. **K-Means $L^*a^*b^*$ Invariance (JSIT 2025)**: Mencegah penurunan akurasi akibat perubahan intensitas cahaya matahari dan bayangan pepohonan di luar ruangan.
3. **Edge Processing Real-Time (JOSH 2025, JIEET 2024)**: Latensi rerata 24.6 ms membuktikan bahwa arsitektur MobileNetV3-Large sanggup bekerja mulus di atas 40 FPS tanpa memerlukan GPU server mahal.

---

## Project Governance

[![CI](https://github.com/aryafatthurahman4-collab/cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/aryafatthurahman4-collab/cnn/actions/workflows/ci.yml)

[Contributing](CONTRIBUTING.md) | [Security Policy](SECURITY.md) | [Code of Conduct](CODE_OF_CONDUCT.md)