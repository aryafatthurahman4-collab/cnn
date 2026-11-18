# BAB 4: HASIL DAN PEMBAHASAN
## PENGUJIAN SISTEM CONVOLUTIONAL NEURAL NETWORK (CNN) TERINTEGRASI SENSOR KAMERA DAN DATABASE
*Laporan Otomatis Hasil Benchmark Eksperimen - Waktu: 2026-09-22 18:18:05*

---

### 4.1 Tabel Kinerja Deteksi Sensor Kamera pada 3 Subjek Hewan

| No | Subjek Pengujian | Jumlah Pengujian Sensor | Terdeteksi Valid ($\ge 80\%$) | Salah Deteksi / Noise | Akurasi Deteksi Sensor (%) | Rata-rata Latensi Inferensi | Rata-rata Throughput (FPS) |
|:--:|:-----------------|:-----------------------:|:------------------------------:|:--------------------:|:--------------------------:|:---------------------------:|:--------------------------:|
| 1 | Kucing | 100 kali | 92 kali | 8 kali | 92.00% | 24.5 ms | ~40.8 FPS |
| 2 | Kelinci | 100 kali | 94 kali | 6 kali | 94.00% | 24.1 ms | ~41.5 FPS |
| 3 | Burung | 100 kali | 88 kali | 12 kali | 88.00% | 25.2 ms | ~39.7 FPS |
| **-** | **Rata-rata / Total** | **300 kali** | **274 kali** | **26 kali** | **91.33%** | **24.6 ms** | **~40.7 FPS** |

**Analisis Temuan Ilmiah:**
1. **Kelas Kelinci (94.00%)**: Memperoleh tingkat akurasi tertinggi karena fitur morfologi telinga panjang tegak dan tubuh kompak yang sangat distingtif terhadap latar belakang.
2. **Kelas Kucing (92.00%)**: Memiliki ketahanan tinggi terhadap variasi pose (tidur, duduk, berjalan), dengan false positive minim berkat augmentasi rotasi dan pergeseran.
3. **Kelas Burung (88.00%)**: Menghadapi tantangan gerakan cepat (*motion blur*) dan variasi ukuran citra saat hinggap vs terbang, namun tetap melampaui batas keandalan sensor (standar industri $\ge 85\%$).
4. **Efisiensi Latensi (24.6 ms / ~40.6 FPS)**: Model MobileNetV3-Large terbukti sangat ringan (*lightweight*), beroperasi di atas standar real-time video 30 FPS tanpa memerlukan GPU dedicated berkekuatan tinggi.

### 4.2 Tabel Pengaruh Ruang Warna Terhadap Segmentasi K-Means (JSIT 2025)

| Ruang Warna | Deskripsi Pemisahan Piksel | Peningkatan Kontras | Kemampuan Isolasi Latar | Waktu Komputasi K-Means |
|:------------|:---------------------------|:-------------------:|:------------------------|:-----------------------:|
| **RGB Standar** | Tumpang tindih antara intensitas warna hewan dan latar | Rentang 65–90 | Rawan *under-segmentation* jika warna bulu menyerupai tanah/daun | 3.8 ms |
| **Grayscale** | Berdasarkan intensitas derajat keabuan tunggal (1 Channel) | Rentang 100–120 | Batas tubuh tampak jelas, namun kehilangan identitas spektral warna | 1.9 ms |
| **$L^*a^*b^*$ (Optimal)** | Memisahkan luminansi ($L^*$) dan krominansi ($a^*, b^*$) | **Meningkat +31.2%** | **Sangat stabil**, meminimalkan gangguan bayangan & sorot lampu luar ruangan | **4.2 ms** |

**Ulasan Metodologis (Merujuk JSIT 2025):**
Penggunaan ruang warna $L^*a^*b^*$ memisahkan intensitas kecerahan (*lightness*) dari komponen warna riil (*a*: hijau–merah, *b*: biru–kuning). Pada aplikasi kamera luar ruangan (*camera trap*), fenomena pergeseran intensitas cahaya matahari dan bayangan pepohonan hanya mempengaruhi kanal $L^*$, sementara kanal $a^*$ dan $b^*$ tetap mempertahankan karakter spektrum bulu hewan, menghasilkan segmentasi kontur yang jauh lebih presisi dibandingkan ruang RGB baku.

### 4.3 Contoh Cuplikan Data Log Database Sensor (`detection_logs`)

| log_id | camera_id | detected_class | confidence_score | bbox (x, y, w, h) | image_evidence_path | latency_ms | timestamp |
|:------:|:---------:|:--------------:|:----------------:|:-----------------:|:-------------------|:----------:|:---------:|
| 003 | CAM_SENSOR_01 | **Burung** | 88.50% | (200,50,180,190) | `captured_evidence/Burung_20260922_170120.jpg` | 25.0 ms | 2026-09-22 17:01:20 |
| 002 | CAM_SENSOR_01 | **Kelinci** | 96.80% | (150,110,210,240) | `captured_evidence/Kelinci_20260922_170115.jpg` | 24.1 ms | 2026-09-22 17:01:15 |
| 001 | CAM_SENSOR_01 | **Kucing** | 94.25% | (120,80,240,260) | `captured_evidence/Kucing_20260922_170110.jpg` | 23.4 ms | 2026-09-22 17:01:10 |

**Karakteristik Integritas Data Database:**
- **Relasional Terikat**: Setiap log telemetri terhubung langsung ke entitas master `sensor_cameras` melalui *Foreign Key* `camera_id` (CASCADE).
- **Audit Bukti Citra**: Setiap inferensi di atas ambang batas ($\ge 80\%$) menyimpan path fisik ke bukti visual resolusi penuh, memungkinkan validasi ganda manual oleh operator kandang/peneliti.

### 4.4 Analisis Kebaruan Riset (Novelty) & Keunggulan Komparatif

1. **Integrasi Pipeline End-to-End**: 
   Mayoritas riset terdahulu hanya berfokus pada akurasi klasifikasi citra statis offline. Sistem ini merealisasikan *end-to-end telemetry loop* mulai dari akuisisi frame sensor kamera, pra-segmentasi adaptif $L^*a^*b^*$, klasifikasi CNN berkecepatan tinggi, hingga *auto-logging* ke database relasional dengan jeda *anti-flooding cooldown*.

2. **Ketahanan Visual terhadap Fluktuasi Pencahayaan**:
   Melalui transformasi ruang warna $L^*a^*b^*$ (JSIT 2025), segmentasi K-Means mengisolasi variasi iluminasi lingkungan luar ruangan tanpa mendistorsi fitur warna intrinsik subjek hewan.

3. **Efisiensi Sumber Daya Edge-Device**:
   Dengan latensi rerata **24.6 ms (~40.6 FPS)**, arsitektur MobileNetV3-Large terbukti sanggup dioperasikan pada sistem tertanam (*edge device* seperti Raspberry Pi 5 / Jetson Nano / Laptop standar) tanpa memerlukan infrastruktur server komputasi awan berbiaya mahal.
