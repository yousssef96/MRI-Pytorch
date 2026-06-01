# 🧠 Brain Tumor Classification

A deep learning project for binary classification of brain MRI scans — detecting the presence or absence of a brain tumor using PyTorch, PyTorch Lightning, and transfer learning with ResNet50.

---

## 📌 Overview

This project builds and compares multiple convolutional neural network architectures trained on a labeled dataset of brain MRI images. It progresses from a custom CNN baseline to a fine-tuned ResNet50 model.


---

## 🗂️ Dataset

The dataset consists of MRI brain scan images organized into two classes:

- **`yes/`** — MRI scans showing a brain tumor
- **`no/`** — MRI scans with no tumor present

Images vary in size and number of channels, so preprocessing includes resizing, cropping, grayscale normalization, and augmentation.

---

## 🏗️ Model Architecture

Three models were trained and evaluated:

| Model | Description | Test Accuracy |
|---|---|---|
| **Custom CNN** | 4-block convolutional network with BatchNorm, Dropout, and AdaptiveAvgPool | ~58% |
| **ResNet50 (frozen)** | Pretrained ImageNet weights, only the classification head trained | Improved |
| **ResNet50 (fine-tuned)** | Last convolutional block (`layer4`) unfrozen for fine-tuning | **~92%** |

---

## ⚙️ Tech Stack

- **PyTorch** — model definition and training
- **PyTorch Lightning** — training loop, callbacks, checkpointing
- **Torchvision** — pretrained ResNet50, image transforms
- **Torchmetrics** — accuracy and confusion matrix
- **scikit-learn** — train/val/test splitting
- **mlxtend** — confusion matrix visualization

---

## 🔄 Training Pipeline

1. Images loaded from `yes/` and `no/` directories
2. Labels assigned (`1` = tumor, `0` = no tumor)
3. 70/15/15 stratified train/val/test split
4. Data augmentation on training set (random crop, rotation, affine)
5. Model trained with `BCEWithLogitsLoss` and `ModelCheckpoint` (best val accuracy saved)
6. Best checkpoint evaluated on the held-out test set

---


## 📊 Results

The fine-tuned ResNet50 model significantly outperforms the custom CNN, demonstrating the value of transfer learning on small medical imaging datasets. Training curves and confusion matrices are generated inline in the notebook.

