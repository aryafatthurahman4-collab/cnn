import os
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Kucing", "Kelinci", "Burung", "Ayam"]
BASE_DIR = "dataset_hewan"

def setup_dataset_structure():
    """
    Membuat struktur direktori dataset untuk 4 kelas hewan: Kucing, Kelinci, Burung, Ayam
    """
    for c in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, c), exist_ok=True)
    print(f"[INFO] Direktori dataset siap di '{BASE_DIR}/': {CLASSES}")

def generate_synthetic_samples(num_per_class=30):
    """
    Membuat citra sampel sintetis awal jika dataset nyata belum dimasukkan.
    Mendukung 4 kelas: Kucing, Kelinci, Burung, Ayam.
    Ini memastikan pipeline training TensorFlow/Keras & PyTorch serta benchmark dapat diverifikasi langsung.
    """
    setup_dataset_structure()
    
    # Karakteristik visual khas masing-masing kelas (warna, latar, bentuk)
    class_styles = {
        "Kucing": {"bg": (220, 210, 190), "fg": (180, 100, 40), "shape": "cat"},
        "Kelinci": {"bg": (200, 230, 200), "fg": (240, 240, 240), "shape": "rabbit"},
        "Burung": {"bg": (180, 220, 250), "fg": (40, 120, 220), "shape": "bird"},
        "Ayam": {"bg": (245, 230, 210), "fg": (180, 70, 20), "shape": "chicken"}
    }

    for c in CLASSES:
        folder = os.path.join(BASE_DIR, c)
        existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing >= 15:
            print(f"[INFO] Kelas '{c}' sudah memiliki {existing} citra. Melewati pembuatan sintetis.")
            continue

        style = class_styles[c]
        print(f"[INFO] Menghasilkan {num_per_class} citra sintetis untuk kelas: {c}...")

        for i in range(1, num_per_class + 1):
            img = Image.new("RGB", (224, 224), style["bg"])
            draw = ImageDraw.Draw(img)

            # Tambahkan variasi noise latar belakang (simulasi kondisi kamera lapangan)
            noise = np.random.randint(-25, 25, (224, 224, 3), dtype=np.int16)
            arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            # Variasi posisi subjek hewan
            offset_x = np.random.randint(-20, 20)
            offset_y = np.random.randint(-20, 20)
            fg = style["fg"]

            if style["shape"] == "cat":
                # Badan & kepala kucing + telinga lancip segitiga
                draw.ellipse([70+offset_x, 80+offset_y, 150+offset_x, 170+offset_y], fill=fg)
                draw.polygon([(70+offset_x, 90+offset_y), (85+offset_x, 40+offset_y), (105+offset_x, 80+offset_y)], fill=(150, 80, 30))
                draw.polygon([(115+offset_x, 80+offset_y), (135+offset_x, 40+offset_y), (150+offset_x, 90+offset_y)], fill=(150, 80, 30))
                # Mata kucing kehijauan & kumis
                draw.ellipse([90+offset_x, 100+offset_y, 100+offset_x, 115+offset_y], fill=(0, 220, 100))
                draw.ellipse([120+offset_x, 100+offset_y, 130+offset_x, 115+offset_y], fill=(0, 220, 100))
                draw.line([(85+offset_x, 125+offset_y), (55+offset_x, 120+offset_y)], fill=(80, 50, 30), width=2)
                draw.line([(135+offset_x, 125+offset_y), (165+offset_x, 120+offset_y)], fill=(80, 50, 30), width=2)

            elif style["shape"] == "rabbit":
                # Tubuh bulat kompak & telinga tegak panjang
                draw.ellipse([60+offset_x, 90+offset_y, 160+offset_x, 185+offset_y], fill=fg)
                # Telinga panjang khas kelinci
                draw.ellipse([80+offset_x, 20+offset_y, 105+offset_x, 100+offset_y], fill=(245, 210, 220))
                draw.ellipse([115+offset_x, 20+offset_y, 140+offset_x, 100+offset_y], fill=(245, 210, 220))
                # Mata merah / hitam khas kelinci
                draw.ellipse([85+offset_x, 105+offset_y, 95+offset_x, 115+offset_y], fill=(180, 30, 30))
                draw.ellipse([125+offset_x, 105+offset_y, 135+offset_x, 115+offset_y], fill=(180, 30, 30))

            elif style["shape"] == "bird":
                # Tubuh ramping bipedal, sayap, paruh lancip
                draw.ellipse([70+offset_x, 70+offset_y, 150+offset_x, 155+offset_y], fill=fg)
                # Paruh kuning
                draw.polygon([(145+offset_x, 95+offset_y), (185+offset_x, 105+offset_y), (145+offset_x, 115+offset_y)], fill=(255, 190, 0))
                # Sayap
                draw.ellipse([80+offset_x, 90+offset_y, 130+offset_x, 145+offset_y], fill=(20, 70, 170))
                # Mata
                draw.ellipse([130+offset_x, 85+offset_y, 140+offset_x, 95+offset_y], fill=(10, 10, 10))

            elif style["shape"] == "chicken":
                # Ayam: Badan gemuk, sayap, jengger merah menyala (comb) & pial (wattle), paruh kuning
                draw.ellipse([65+offset_x, 80+offset_y, 155+offset_x, 165+offset_y], fill=fg)
                # Kepala
                draw.ellipse([120+offset_x, 60+offset_y, 165+offset_x, 105+offset_y], fill=(210, 90, 30))
                # Jengger Merah (Comb) di atas kepala
                draw.polygon([(135+offset_x, 65+offset_y), (140+offset_x, 35+offset_y), (150+offset_x, 50+offset_y), (158+offset_x, 38+offset_y), (165+offset_x, 65+offset_y)], fill=(230, 20, 20))
                # Pial Merah (Wattle) di bawah paruh
                draw.ellipse([155+offset_x, 100+offset_y, 168+offset_x, 120+offset_y], fill=(230, 20, 20))
                # Paruh lancip kuning
                draw.polygon([(160+offset_x, 80+offset_y), (188+offset_x, 90+offset_y), (160+offset_x, 98+offset_y)], fill=(255, 180, 0))
                # Sayap cokelat tua
                draw.ellipse([75+offset_x, 95+offset_y, 130+offset_x, 150+offset_y], fill=(140, 50, 10))
                # Mata
                draw.ellipse([145+offset_x, 75+offset_y, 153+offset_x, 83+offset_y], fill=(20, 20, 20))

            filename = os.path.join(folder, f"sample_{c}_{i:03d}.jpg")
            img.save(filename, quality=92)

    print(f"[INFO] Dataset 4 kelas siap di direktori '{BASE_DIR}'.")

def inspect_dataset():
    """
    Menampilkan jumlah citra untuk setiap kelas hewan
    """
    print("="*60)
    print("STATUS DATASET DETEKSI HEWAN CNN & YOLOV8")
    print("="*60)
    total = 0
    for c in CLASSES:
        folder = os.path.join(BASE_DIR, c)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f" - Kelas '{c:<8}': {len(files):>4} citra")
            total += len(files)
        else:
            print(f" - Kelas '{c:<8}': TIDAK DITEMUKAN")
    print("-" * 60)
    print(f" Total Seluruh Citra: {total} citra")
    print("="*60)

if __name__ == "__main__":
    generate_synthetic_samples(num_per_class=30)
    inspect_dataset()
