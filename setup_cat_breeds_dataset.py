"""
Setup Dataset 12 Ras Kucing - Referensi JIEET 2024
Ras: Abyssinia, Bengal, Bombay, Birman, British Shorthair, Egyptian, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import shutil

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

BASE_DIR = "dataset_cat_breeds"

def setup_cat_breeds_structure():
    """Membuat struktur direktori untuk 12 ras kucing"""
    for breed in CAT_BREEDS:
        breed_dir = os.path.join(BASE_DIR, breed)
        os.makedirs(breed_dir, exist_ok=True)
    print(f"[INFO] Direktori dataset 12 ras kucing siap di '{BASE_DIR}/'")
    print(f"[INFO] Total ras: {len(CAT_BREEDS)}")
    for breed in CAT_BREEDS:
        print(f"  - {breed}")

def generate_cat_breed_samples(num_per_class=20):
    """
    Membuat citra sampel sintetis untuk 12 ras kucing
    Karakteristik visual berbeda untuk setiap ras sesuai referensi
    """
    setup_cat_breeds_structure()
    
    # Karakteristik visual untuk setiap ras kucing
    breed_styles = {
        "Abyssinia": {
            "bg": (210, 200, 180),
            "fg": (194, 162, 132),
            "pattern": "ticked",
            "ear_shape": "pointed",
            "eye_color": (140, 180, 100)
        },
        "Bengal": {
            "bg": (220, 210, 190),
            "fg": (180, 140, 90),
            "pattern": "spotted",
            "ear_shape": "rounded",
            "eye_color": (30, 150, 200)
        },
        "Bombay": {
            "bg": (200, 200, 200),
            "fg": (30, 30, 30),
            "pattern": "solid",
            "ear_shape": "rounded",
            "eye_color": (200, 150, 50)
        },
        "Birman": {
            "bg": (230, 230, 230),
            "fg": (240, 240, 240),
            "pattern": "colorpoint",
            "ear_shape": "rounded",
            "eye_color": (30, 150, 255)
        },
        "British Shorthair": {
            "bg": (200, 200, 200),
            "fg": (100, 150, 180),
            "pattern": "solid",
            "ear_shape": "rounded",
            "eye_color": (200, 150, 100)
        },
        "Egyptian": {
            "bg": (220, 200, 180),
            "fg": (200, 180, 140),
            "pattern": "spotted",
            "ear_shape": "large_pointed",
            "eye_color": (30, 200, 50)
        },
        "Maine Coon": {
            "bg": (210, 200, 190),
            "fg": (160, 130, 100),
            "pattern": "tabby",
            "ear_shape": "tufted",
            "eye_color": (30, 150, 200)
        },
        "Persian": {
            "bg": (230, 230, 230),
            "fg": (240, 240, 240),
            "pattern": "solid",
            "ear_shape": "small_rounded",
            "eye_color": (30, 150, 255)
        },
        "Ragdoll": {
            "bg": (230, 230, 230),
            "fg": (250, 250, 250),
            "pattern": "colorpoint",
            "ear_shape": "rounded",
            "eye_color": (30, 150, 255)
        },
        "Russian Blue": {
            "bg": (200, 200, 200),
            "fg": (100, 150, 180),
            "pattern": "solid",
            "ear_shape": "pointed",
            "eye_color": (30, 200, 100)
        },
        "Siamese": {
            "bg": (230, 230, 230),
            "fg": (220, 200, 180),
            "pattern": "colorpoint",
            "ear_shape": "large_pointed",
            "eye_color": (30, 150, 255)
        },
        "Sphynx": {
            "bg": (220, 210, 200),
            "fg": (200, 180, 160),
            "pattern": "hairless",
            "ear_shape": "large_pointed",
            "eye_color": (30, 200, 100)
        }
    }

    for breed in CAT_BREEDS:
        breed_dir = os.path.join(BASE_DIR, breed)
        existing = len([f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing >= 5:
            print(f"[INFO] Ras '{breed}' sudah memiliki {existing} citra. Melewati pembuatan sintetis.")
            continue

        style = breed_styles[breed]
        print(f"[INFO] Menghasilkan {num_per_class} citra sintetis untuk ras: {breed}...")

        for i in range(1, num_per_class + 1):
            img = Image.new("RGB", (224, 224), style["bg"])
            draw = ImageDraw.Draw(img)

            # Noise latar belakang
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            # Offset posisi untuk variasi
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            fg = style["fg"]
            pattern = style["pattern"]
            ear_shape = style["ear_shape"]
            eye_color = style["eye_color"]

            # Badan kucing
            body_center_x = 112 + offset_x
            body_center_y = 140 + offset_y
            body_width = 70
            body_height = 60
            
            draw.ellipse([body_center_x - body_width//2, body_center_y - body_height//2,
                         body_center_x + body_width//2, body_center_y + body_height//2], fill=fg)

            # Kepala kucing
            head_center_x = 112 + offset_x
            head_center_y = 90 + offset_y
            head_width = 55
            head_height = 50
            
            draw.ellipse([head_center_x - head_width//2, head_center_y - head_height//2,
                         head_center_x + head_width//2, head_center_y + head_height//2], fill=fg)

            # Telinga berdasarkan ras
            if ear_shape == "pointed":
                # Telinga lancip standar
                draw.polygon([(head_center_x - 20 + offset_x, head_center_y - 15 + offset_y),
                             (head_center_x - 10 + offset_x, head_center_y - 40 + offset_y),
                             (head_center_x + 5 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
                draw.polygon([(head_center_x + 20 + offset_x, head_center_y - 15 + offset_y),
                             (head_center_x + 10 + offset_x, head_center_y - 40 + offset_y),
                             (head_center_x - 5 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
            elif ear_shape == "large_pointed":
                # Telinga besar lancip (Siamese, Egyptian)
                draw.polygon([(head_center_x - 25 + offset_x, head_center_y - 10 + offset_y),
                             (head_center_x - 15 + offset_x, head_center_y - 50 + offset_y),
                             (head_center_x + 0 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
                draw.polygon([(head_center_x + 25 + offset_x, head_center_y - 10 + offset_y),
                             (head_center_x + 15 + offset_x, head_center_y - 50 + offset_y),
                             (head_center_x + 0 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
            elif ear_shape == "rounded":
                # Telinga bulat (British Shorthair, Bombay)
                draw.ellipse([head_center_x - 30 + offset_x, head_center_y - 35 + offset_y,
                            head_center_x - 10 + offset_x, head_center_y - 10 + offset_y], fill=fg)
                draw.ellipse([head_center_x + 10 + offset_x, head_center_y - 35 + offset_y,
                            head_center_x + 30 + offset_x, head_center_y - 10 + offset_y], fill=fg)
            elif ear_shape == "tufted":
                # Telinga dengan rumbai (Maine Coon)
                draw.polygon([(head_center_x - 25 + offset_x, head_center_y - 10 + offset_y),
                             (head_center_x - 15 + offset_x, head_center_y - 45 + offset_y),
                             (head_center_x + 0 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
                draw.polygon([(head_center_x + 25 + offset_x, head_center_y - 10 + offset_y),
                             (head_center_x + 15 + offset_x, head_center_y - 45 + offset_y),
                             (head_center_x + 0 + offset_x, head_center_y - 15 + offset_y)], fill=fg)
                # Tambah rumbai kecil
                draw.polygon([(head_center_x - 20 + offset_x, head_center_y - 45 + offset_y),
                             (head_center_x - 18 + offset_x, head_center_y - 55 + offset_y),
                             (head_center_x - 12 + offset_x, head_center_y - 45 + offset_y)], fill=(150, 130, 100))
                draw.polygon([(head_center_x + 20 + offset_x, head_center_y - 45 + offset_y),
                             (head_center_x + 18 + offset_x, head_center_y - 55 + offset_y),
                             (head_center_x + 12 + offset_x, head_center_y - 45 + offset_y)], fill=(150, 130, 100))
            elif ear_shape == "small_rounded":
                # Telinga kecil bulat (Persian)
                draw.ellipse([head_center_x - 25 + offset_x, head_center_y - 25 + offset_y,
                            head_center_x - 12 + offset_x, head_center_y - 10 + offset_y], fill=fg)
                draw.ellipse([head_center_x + 12 + offset_x, head_center_y - 25 + offset_y,
                            head_center_x + 25 + offset_x, head_center_y - 10 + offset_y], fill=fg)

            # Mata
            eye_size = 8
            draw.ellipse([head_center_x - 15 + offset_x, head_center_y - 5 + offset_y,
                         head_center_x - 15 + eye_size + offset_x, head_center_y - 5 + eye_size + offset_y], fill=eye_color)
            draw.ellipse([head_center_x + 7 + offset_x, head_center_y - 5 + offset_y,
                         head_center_x + 7 + eye_size + offset_x, head_center_y - 5 + eye_size + offset_y], fill=eye_color)

            # Pola bulu
            if pattern == "spotted":
                # Bintik-bintik (Bengal, Egyptian)
                for _ in range(8):
                    spot_x = np.random.randint(body_center_x - 30, body_center_x + 30)
                    spot_y = np.random.randint(body_center_y - 25, body_center_y + 25)
                    spot_size = np.random.randint(3, 8)
                    draw.ellipse([spot_x, spot_y, spot_x + spot_size, spot_y + spot_size], fill=(140, 100, 60))
            elif pattern == "tabby":
                # Garis-garis (Maine Coon)
                for j in range(5):
                    line_y = body_center_y - 20 + j * 10
                    draw.line([body_center_x - 30 + offset_x, line_y + offset_y,
                              body_center_x + 30 + offset_x, line_y + offset_y], fill=(120, 90, 60), width=2)
            elif pattern == "colorpoint":
                # Colorpoint (Birman, Ragdoll, Siamese) - gelap di ujung
                draw.ellipse([head_center_x - head_width//2, head_center_y + head_height//2 - 10,
                            head_center_x + head_width//2, head_center_y + head_height//2 + 5], fill=(180, 160, 140))
                draw.ellipse([body_center_x - body_width//2, body_center_y + body_height//2 - 10,
                            body_center_x + body_width//2, body_center_y + body_height//2 + 10], fill=(180, 160, 140))
            elif pattern == "hairless":
                # Kulit terlihat (Sphynx)
                draw.ellipse([head_center_x - 15 + offset_x, head_center_y + 5 + offset_y,
                            head_center_x + 15 + offset_x, head_center_y + 15 + offset_y], fill=(180, 160, 140))

            filename = os.path.join(breed_dir, f"{breed.replace(' ', '_')}_{i:03d}.jpg")
            img.save(filename, quality=90)

    print(f"[INFO] Sampel dataset 12 ras kucing siap di direktori '{BASE_DIR}'.")

def inspect_cat_breeds_dataset():
    """Menampilkan ringkasan dataset 12 ras kucing"""
    print("\n--- RINGKASAN DATASET 12 RAS KUCING ---")
    total = 0
    for breed in CAT_BREEDS:
        breed_dir = os.path.join(BASE_DIR, breed)
        count = len([f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(breed_dir) else 0
        total += count
        print(f" Ras {breed:<20}: {count} citra")
    print(f" Total Keseluruhan: {total} citra")
    print(f" Target sesuai JIEET 2024: 2.402 citra (untuk 12 ras)\n")

if __name__ == "__main__":
    generate_cat_breed_samples(num_per_class=25)
    inspect_cat_breeds_dataset()
