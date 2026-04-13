# Solar Panel Fault Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-MobileNetV2-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-77.01%25-success?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

Automated visual inspection of solar panels using Transfer Learning (MobileNetV2), classifying panel images into six fault categories to enable scalable, maintenance-driven monitoring without manual intervention.

---

## Overview

Solar panels degrade silently. Dust accumulation, bird droppings, physical cracks, snow coverage, and electrical damage all reduce energy output — yet spotting them manually across large solar farms is slow, expensive, and inconsistent.

This project builds a deep learning image classifier capable of automatically detecting these faults from photographs. It leverages MobileNetV2 pre-trained on ImageNet with a custom classification head, fine-tuned on a labelled solar panel dataset using a two-phase transfer learning strategy.

> **Academic Context:** Deep Learning — IA2 | AI & DS TY SEM VI | K. J. Somaiya School of Engineering  
> **Authors:** Aviral Goel (16014223102) & Sachi Parekh (16014223069)

---

## Problem Statement

Given an image of a solar panel, predict which of the following six conditions it belongs to:

| Class | Description |
|---|---|
| Bird-drop | Panel surface contaminated with bird droppings |
| Clean | Panel is clean and operating normally |
| Dusty | Surface contaminated with dust |
| Electrical-damage | Visible electrical faults such as burn marks or damaged wiring |
| Physical-Damage | Structural damage including cracks or chips |
| Snow-Covered | Panel partially or fully covered with snow |

---

## Repository Structure

```
solar-panel-fault-detection/
|
|-- dataset/
|   |-- Bird-drop/
|   |-- Clean/
|   |-- Dusty/
|   |-- Electrical-damage/
|   |-- Physical-Damage/
|   +-- Snow-Covered/
|
|-- notebooks/
|   +-- solar_fault_detection.ipynb
|
|-- models/
|   +-- best_model.h5
|
|-- outputs/
|   |-- classification_report.txt
|   |-- confusion_matrix.png
|   +-- training_curves.png
|
|-- requirements.txt
+-- README.md
```

---

## Dataset

**Source:** [Solar Panel Images - Clean & Faulty](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images-clean-faulty) by `pythonafroz` on Kaggle

| Class | Image Count |
|---|---|
| Bird-drop | 191 |
| Clean | 193 |
| Dusty | 190 |
| Electrical-damage | 103 |
| Physical-Damage | 69 |
| Snow-Covered | 123 |
| **Total** | **869** |

The dataset is moderately imbalanced. `Physical-Damage` is the most underrepresented class with only 69 images, which directly impacts recall for that class during evaluation.

---

## Model Architecture

**Backbone:** MobileNetV2 pre-trained on ImageNet (`include_top=False`)

MobileNetV2 uses depthwise separable convolutions and inverted residual blocks, making it computationally efficient while retaining strong feature extraction capability. This makes it well-suited for constrained environments and potential edge deployment.

**Custom Classification Head:**

```
MobileNetV2 Backbone
        |
GlobalAveragePooling2D          # Reduces 7x7x1280 feature maps to a 1280-dim vector
        |
Dense(256) + BatchNorm + ReLU + Dropout(0.4) + L2(1e-4)
        |
Dense(128) + BatchNorm + ReLU + Dropout(0.3) + L2(1e-4)
        |
Dense(6, softmax)               # 6-class probability output
```

---

## Preprocessing and Augmentation

All images were resized to 224x224 pixels and normalized by rescaling pixel values to the [0, 1] range.

Training set augmentations applied via Keras `ImageDataGenerator`:

| Technique | Configuration | Purpose |
|---|---|---|
| Rotation | +/- 20 degrees | Simulate different capture angles |
| Width / Height Shift | 15% | Random translation |
| Shear | 10% | Perspective distortion |
| Zoom | +/- 20% | Capture close-up and wide perspectives |
| Brightness | [0.7, 1.3] | Simulate varying lighting conditions |
| Horizontal Flip | Enabled | Mirror augmentation |
| Vertical Flip | Disabled | Panels are always upright |

Validation data received rescaling only — no augmentation — to ensure unbiased evaluation. An 80/20 train-validation split was applied.

---

## Training Strategy

A two-phase transfer learning strategy was employed to maximise the use of pre-trained ImageNet features while adapting them to solar panel imagery.

### Phase 1 — Feature Extraction (Epochs 1-15)

- MobileNetV2 backbone fully frozen; only the classification head is trained
- Optimizer: Adam with learning rate 1e-3
- Allows the new layers to converge before any backbone weights are modified

### Phase 2 — Fine-Tuning (Epochs 16-35)

- Last 30 layers of MobileNetV2 unfrozen
- Optimizer: Adam with reduced learning rate 1e-5 to avoid destroying pre-trained features
- Earlier layers remain frozen as they encode general low-level features

### Regularization and Callbacks

| Technique | Configuration |
|---|---|
| L2 Weight Decay | 1e-4 on both Dense layers |
| Dropout | 0.4 after Dense(256), 0.3 after Dense(128) |
| BatchNormalization | After each Dense layer |
| EarlyStopping | Patience = 5 (Phase 1), Patience = 7 (Phase 2) |
| ReduceLROnPlateau | Factor = 0.3 on validation loss plateau |
| ModelCheckpoint | Saves best model by validation accuracy |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |

---

## Results

### Overall Metrics

| Metric | Score |
|---|---|
| Overall Accuracy | 77.01% |
| Weighted Precision | 79.30% |
| Weighted Recall | 77.01% |
| Weighted F1-Score | 76.92% |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bird-drop | 0.75 | 0.66 | 0.70 |
| Clean | 0.65 | 0.97 | 0.78 |
| Dusty | 0.74 | 0.66 | 0.69 |
| Electrical-damage | 0.93 | 0.70 | 0.80 |
| Physical-Damage | 1.00 | 0.69 | 0.82 |
| Snow-Covered | 0.96 | 0.92 | 0.94 |
| **Weighted Average** | **0.79** | **0.77** | **0.77** |

---

## Results Analysis

- The two-phase training strategy proved effective. Head stabilisation in Phase 1, followed by targeted backbone fine-tuning in Phase 2, enabled meaningful feature adaptation from ImageNet to solar panel imagery.
- Snow-Covered achieved the highest F1-score (0.94) owing to its visually distinct appearance — uniform white coverage is easily separable from all other classes.
- Physical-Damage achieved perfect precision (1.00), meaning every prediction made for this class was correct. However, recall of 0.69 indicates some instances were missed, likely due to the very small training sample of 69 images.
- Electrical-damage performed well (F1: 0.80) with high precision (0.93), confirming the model reliably identifies visible electrical faults.
- The Clean class shows high recall (0.97) but lower precision (0.65), indicating the model frequently predicts Clean for visually ambiguous images.
- Dusty and Bird-drop classes show the weakest performance (F1: 0.69 and 0.70) due to subtle visual differences from the Clean class.
- Multiple regularization techniques — Dropout, BatchNormalization, L2 weight decay, EarlyStopping, and ReduceLROnPlateau — collectively helped limit overfitting across both training phases.

---

## Limitations

- The dataset is small (869 images total), which constrains the model's ability to generalise to diverse real-world conditions.
- Severe class imbalance, particularly for Physical-Damage (69 images), limits recall despite good precision.
- The model classifies individual images independently and does not capture temporal degradation patterns across panel lifespans.
- Generalisation to different panel hardware types, orientations, or lighting conditions has not been validated.
- MobileNetV2 may underperform larger backbones such as EfficientNetB3 or ResNet50 on complex fault types with subtle visual cues.

---

## Future Work

- Upgrade backbone to EfficientNetB3 or a Vision Transformer (ViT) for greater representational capacity.
- Apply Grad-CAM visualisation to identify which regions of a solar panel drive model predictions.
- Implement Test-Time Augmentation (TTA) to improve inference robustness.
- Explore model ensembling (MobileNetV2 + ResNet50) for additional accuracy gains.
- Collect more labelled data for underrepresented classes, particularly Physical-Damage and Electrical-damage.
- Deploy the model as a REST API using FastAPI and integrate it with a Streamlit dashboard for real-time field monitoring.

---

## Setup and Usage

**1. Clone the repository**

```bash
git clone https://github.com/goelavi04/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from Kaggle: [Solar Panel Images - Clean & Faulty](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images-clean-faulty)  
Place the class folders inside the `dataset/` directory as shown in the repository structure above.

**4. Run the notebook**

```bash
jupyter notebook notebooks/solar_fault_detection.ipynb
```

---

## Requirements

```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
jupyter
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## References

- Howard, A. G., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
- Kaggle Dataset: [pythonafroz/solar-panel-images-clean-faulty](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images-clean-faulty)
- Keras Documentation: https://keras.io
- TensorFlow Documentation: https://www.tensorflow.org

---

## Authors

| Name | Roll Number | Institute |
|---|---|---|
| Aviral Goel | 16014223102 | K. J. Somaiya School of Engineering |
| Sachi Parekh | 16014223069 | K. J. Somaiya School of Engineering |

