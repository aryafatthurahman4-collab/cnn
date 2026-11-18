"""
Training ResNet50V2 untuk Klasifikasi 12 Ras Kucing
Referensi: JIEET Volume 08 Nomor 01, 2024
Arsitektur: ResNet50V2 dengan Transfer Learning
Optimizer: SGD (Stochastic Gradient Descent)
Dataset: 12 Ras Kucing (Abyssinia, Bengal, Bombay, Birman, British Shorthair, Egyptian, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Konfigurasi sesuai JIEET 2024
DATASET_DIR = "dataset_cat_breeds"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Sesuai JIEET 2024
LEARNING_RATE = 0.01  # Learning rate untuk SGD
MOMENTUM = 0.9  # Momentum untuk SGD

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

print("=" * 80)
print("TRAINING RESNET50V2 - KLASIFIKASI 12 RAS KUCING")
print("Referensi: JIEET Volume 08 Nomor 01, 2024")
print("=" * 80)

# 1. Pipeline Augmentasi Data (80% training, 20% validation sesuai JIEET 2024)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% train, 20% validation
)

print("[INFO] Memuat data latihan dan validasi...")
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CAT_BREEDS,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CAT_BREEDS,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"[INFO] Training samples: {train_gen.samples}")
print(f"[INFO] Validation samples: {val_gen.samples}")
print(f"[INFO] Classes: {len(CAT_BREEDS)}")

# 2. Membangun Arsitektur ResNet50V2 dengan Transfer Learning
print("[INFO] Membangun Arsitektur ResNet50V2 Transfer Learning...")

# Load ResNet50V2 pre-trained on ImageNet
base_model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze backbone layers
base_model.trainable = False

# Tambahkan custom head untuk 12 kelas
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Dense layer dengan 256 unit + ReLU + Dropout 0.3 (sesuai JIEET 2024)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

# Dense layer dengan 128 unit + ReLU + Dropout 0.3 (sesuai JIEET 2024)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

# Output layer dengan 12 unit untuk 12 ras kucing + Softmax
outputs = Dense(len(CAT_BREEDS), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile dengan SGD optimizer (sesuai JIEET 2024)
optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# 3. Callbacks untuk training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_cat_breeds_resnet50v2.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# 4. Proses Training
print(f"[INFO] Memulai training model ({EPOCHS} epochs)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 5. Simpan Model Akhir
model.save("cat_breeds_resnet50v2_final.h5")
print("[INFO] Model final disimpan ke 'cat_breeds_resnet50v2_final.h5'")

# 6. Visualisasi Training History
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy (ResNet50V2)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss (ResNet50V2)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('cat_breeds_resnet50v2_training_curve.png', dpi=300, bbox_inches='tight')
print("[INFO] Grafik training disimpan ke 'cat_breeds_resnet50v2_training_curve.png'")

# 7. Evaluasi Final
print("\n" + "=" * 80)
print("HASIL EVALUASI FINAL")
print("=" * 80)
train_loss, train_acc = model.evaluate(train_gen, verbose=0)
val_loss, val_acc = model.evaluate(val_gen, verbose=0)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

print("\nTarget JIEET 2024:")
print(f"Training Accuracy: 99.17%")
print(f"Validation Accuracy: 88.13%")
print(f"Training Loss: 0.3760")
print(f"Validation Loss: 0.7857")

print("\n[INFO] Training selesai!")
