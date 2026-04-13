"""
Solar Panel Surface Defect Classification
Deep Learning IA-2 | 216U42C601 | AI & DS TY SEM VI
K. J. Somaiya School of Engineering

Model: MobileNetV2 (Transfer Learning)
Classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-Damage, Snow-Covered
"""

# ============================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42

DATASET_PATH = "C:/Users/Aviral Goel/Downloads/archive (8)/Faulty_solar_panel"

NUM_CLASSES = 6

# ============================================================
# SECTION 3: DATA PREPROCESSING & AUGMENTATION
# ============================================================

print("\n" + "="*60)
print("SECTION 3: DATA PREPROCESSING & AUGMENTATION")
print("="*60)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.20,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    validation_split=0.20
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print("\nClass Indices:", train_generator.class_indices)
print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)

# ============================================================
# SECTION 4: DATASET ANALYSIS & VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("SECTION 4: DATASET ANALYSIS")
print("="*60)

class_counts = {}
for cls in os.listdir(DATASET_PATH):
    cls_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(cls_path):
        class_counts[cls] = len(os.listdir(cls_path))

print("\nClass Distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls:25s}: {count} images")

total = sum(class_counts.values())
print(f"\n  Total images: {total}")

classes = list(class_counts.keys())
counts = list(class_counts.values())
colors = ['#2ecc71', '#e67e22', '#8e44ad', '#e74c3c', '#3498db', '#1abc9c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(classes, counts, color=colors, edgecolor='black', linewidth=0.7)
axes[0].set_title('Class Distribution (Image Count)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(counts):
    axes[0].text(i, v + 1, str(v), ha='center', fontsize=10)

axes[1].pie(counts, labels=classes, colors=colors, autopct='%1.1f%%',
            startangle=140, pctdistance=0.8)
axes[1].set_title('Class Distribution (Proportion)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: class_distribution.png")

# Sample images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (cls, color) in enumerate(zip(classes, colors)):
    cls_path = os.path.join(DATASET_PATH, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if imgs:
        img_path = os.path.join(cls_path, imgs[0])
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'Class: {cls}\nShape: {img.shape}', fontsize=11, fontweight='bold')
        axes[idx].axis('off')

plt.suptitle('Sample Images — Solar Panel Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: sample_images.png")

# ============================================================
# SECTION 5: MODEL ARCHITECTURE
# ============================================================

print("\n" + "="*60)
print("SECTION 5: MODEL ARCHITECTURE (MobileNetV2 + Custom Head)")
print("="*60)

def build_model(num_classes=NUM_CLASSES, trainable_base=False):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=trainable_base)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model(trainable_base=False)
model.summary()

# ============================================================
# SECTION 6: TRAINING PHASE 1 — FEATURE EXTRACTION
# ============================================================

print("\n" + "="*60)
print("SECTION 6: PHASE 1 — Feature Extraction (Frozen Base)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3,
                      min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model_phase1.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

print("\nStarting Phase 1 training...")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

# ============================================================
# SECTION 7: TRAINING PHASE 2 — FINE-TUNING
# ============================================================

print("\n" + "="*60)
print("SECTION 7: PHASE 2 — Fine-Tuning (Unfreeze last 30 layers)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Trainable layers in base: {sum(1 for l in base_model.layers if l.trainable)}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3,
                      min_lr=1e-8, verbose=1),
    ModelCheckpoint('best_model_final.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

print("\nStarting Phase 2 fine-tuning...")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============================================================
# SECTION 8: TRAINING CURVES
# ============================================================

print("\n" + "="*60)
print("SECTION 8: TRAINING CURVES")
print("="*60)

def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history1, history2)
phase1_end = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['accuracy'], label='Train Accuracy', color='#2980b9', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
axes[0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
axes[0].set_title('Model Accuracy over Epochs', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['loss'], label='Train Loss', color='#2980b9', linewidth=2)
axes[1].plot(history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
axes[1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
axes[1].set_title('Model Loss over Epochs', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: training_curves.png")

# ============================================================
# SECTION 9: EVALUATION & METRICS
# ============================================================

print("\n" + "="*60)
print("SECTION 9: MODEL EVALUATION")
print("="*60)

best_model = keras.models.load_model('best_model_final.keras')

val_generator.reset()
y_pred_probs = best_model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

class_labels = list(val_generator.class_indices.keys())

overall_accuracy = np.mean(y_pred == y_true)
overall_precision = precision_score(y_true, y_pred, average='weighted')
overall_recall = recall_score(y_true, y_pred, average='weighted')
overall_f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\n{'='*40}")
print(f"  Overall Accuracy  : {overall_accuracy*100:.2f}%")
print(f"  Weighted Precision: {overall_precision*100:.2f}%")
print(f"  Weighted Recall   : {overall_recall*100:.2f}%")
print(f"  Weighted F1-Score : {overall_f1*100:.2f}%")
print(f"{'='*40}")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ============================================================
# SECTION 10: CONFUSION MATRIX
# ============================================================

print("\n" + "="*60)
print("SECTION 10: CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].tick_params(axis='x', rotation=30)

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")

# ============================================================
# SECTION 11: PER-CLASS METRICS BAR CHART
# ============================================================

report_dict = classification_report(y_true, y_pred, target_names=class_labels,
                                     output_dict=True)
metrics_data = {cls: report_dict[cls] for cls in class_labels}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(class_labels))
width = 0.25

ax.bar(x - width, [metrics_data[c]['precision'] for c in class_labels],
       width, label='Precision', color='#3498db', alpha=0.85)
ax.bar(x, [metrics_data[c]['recall'] for c in class_labels],
       width, label='Recall', color='#2ecc71', alpha=0.85)
ax.bar(x + width, [metrics_data[c]['f1-score'] for c in class_labels],
       width, label='F1-Score', color='#e74c3c', alpha=0.85)

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Precision, Recall & F1-Score', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_labels, rotation=20, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: per_class_metrics.png")

# ============================================================
# SECTION 12: SAMPLE PREDICTIONS
# ============================================================

print("\n" + "="*60)
print("SECTION 12: SAMPLE PREDICTIONS")
print("="*60)

val_generator.reset()
sample_images, sample_labels = next(val_generator)
sample_preds = best_model.predict(sample_images[:12], verbose=0)
sample_pred_classes = np.argmax(sample_preds, axis=1)
sample_true_classes = np.argmax(sample_labels[:12], axis=1)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(12):
    axes[i].imshow(sample_images[i])
    pred_cls = class_labels[sample_pred_classes[i]]
    true_cls = class_labels[sample_true_classes[i]]
    confidence = sample_preds[i][sample_pred_classes[i]] * 100
    correct = pred_cls == true_cls
    color = '#2ecc71' if correct else '#e74c3c'
    axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls} ({confidence:.1f}%)",
                       fontsize=9, color=color, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: sample_predictions.png")

# ============================================================
# SECTION 13: SUMMARY
# ============================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"""
Model        : MobileNetV2 (Transfer Learning from ImageNet)
Training     : Phase 1 (Feature Extraction) + Phase 2 (Fine-tuning)
Optimizer    : Adam (LR: 1e-3 -> 1e-5 fine-tune)
Regularization : L2 weight decay, Dropout (0.4, 0.3), BatchNormalization
Augmentation : Rotation, Flip, Zoom, Brightness, Shift, Shear

Results:
  Accuracy  : {overall_accuracy*100:.2f}%
  Precision : {overall_precision*100:.2f}%
  Recall    : {overall_recall*100:.2f}%
  F1-Score  : {overall_f1*100:.2f}%

Saved files:
  class_distribution.png
  sample_images.png
  training_curves.png
  confusion_matrix.png
  per_class_metrics.png
  sample_predictions.png
  best_model_final.keras
""")