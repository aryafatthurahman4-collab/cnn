"""
Script alternatif untuk menjalankan kamera secara manual dengan konfigurasi fleksibel
Gunakan ini jika ingin menguji dengan webcam fisik langsung
"""

import subprocess
import sys

print("=" * 60)
print("PILIH SUMBER KAMERA:")
print("=" * 60)
print("1. Webcam USB / Kamera Laptop (source: 0)")
print("2. Stream Test dari Dataset (source: test)")
print("3. Mode Headless (tanpa tampilan GUI)")
print("=" * 60)

choice = input("Pilih opsi (1-3): ").strip()

if choice == "1":
    print("\n[INFO] Menjalankan kamera webcam...")
    print("[INFO] Tekan 'q' pada jendela video untuk menghentikan")
    subprocess.run([sys.executable, "camera_detector_system.py", "--source", "0"])
elif choice == "2":
    print("\n[INFO] Menjalankan stream test dari dataset...")
    frames = input("Jumlah frame (default 100): ").strip()
    frames_arg = ["--frames", frames] if frames else []
    subprocess.run([sys.executable, "camera_detector_system.py", "--source", "test"] + frames_arg)
elif choice == "3":
    print("\n[INFO] Menjalankan mode headless...")
    frames = input("Jumlah frame (default 100): ").strip()
    frames_arg = ["--frames", frames] if frames else []
    subprocess.run([sys.executable, "camera_detector_system.py", "--source", "test", "--headless"] + frames_arg)
else:
    print("[ERROR] Pilihan tidak valid")
