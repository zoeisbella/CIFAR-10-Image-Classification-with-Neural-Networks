# CIFAR-10 Image Classification Experiment: Using Convolutional Neural Networks (CNN) 🚀

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) model using TensorFlow and Keras for the CIFAR-10 image classification task. CIFAR-10 is a classic benchmark dataset containing 60,000 32x32 pixel color images divided into 10 classes (e.g., airplane, automobile, bird). The project includes data preprocessing, model building, training, evaluation, and visualization, aiming to achieve high classification accuracy.

### 🎯 Experiment Goals
- Build an efficient CNN model to handle small-sized image classification.  
- Optimize the training process with data augmentation and callbacks to avoid overfitting.  
- Analyze model performance across different classes and visualize training history, confusion matrix, and per-class accuracy.

**Final Results**: Test accuracy reaches **87.70%**, with best validation accuracy **87.70%**. 🏆

## 🗂️ Dataset

- **Source**: CIFAR-10 dataset (built-in TensorFlow).  
- **Scale**:  
  - Training set: 50,000 images (32×32×3).  
  - Test set: 10,000 images (32×32×3).  
- **Classes** (10 classes):  
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.  

- **Preprocessing**:  
  - Pixel values normalized to [0, 1] (divided by 255).  
  - Data augmentation: Rotation (15°), translation (0.1), horizontal flip, zoom (0.1), implemented via `ImageDataGenerator`.  

**Example sample images** (from Notebook output):  
<img width="951" height="633" alt="image" src="https://github.com/user-attachments/assets/08a788b4-01d7-4588-8e6d-6b40d662384a" />

*(Note: Running the notebook generates a 5x5 sample grid.)* 🖼️

## 🏗️ Model Architecture Summary

The model is a deep CNN built with Sequential API, featuring 3 convolutional blocks + fully connected layers, with approximately 1.2M parameters. The design is inspired by VGG-like structures, adjusted for CIFAR-10's small images in terms of layer count and regularization.

### 🔧 Architecture Details
| Layer Type                  | Output Shape | Parameters | Description                  |
|-----------------------------|--------------|------------|------------------------------|
| **Conv2D (32 filters, 3x3)** | (32, 32, 32) | ~270      | ReLU activation, same padding |
| **BatchNormalization**      | (32, 32, 32) | ~128      | Normalization                |
| **Conv2D (32 filters, 3x3)** | (32, 32, 32) | ~18K      | ReLU activation, same padding |
| **BatchNormalization**      | (32, 32, 32) | ~128      | Normalization                |
| **MaxPooling2D (2x2)**      | (16, 16, 32) | 0         | Pooling                      |
| **Dropout (0.25)**          | (16, 16, 32) | 0         | Regularization               |
| **Conv2D (64 filters, 3x3)** | (16, 16, 64) | ~18K      | ReLU activation, same padding |
| **BatchNormalization**      | (16, 16, 64) | ~256      | Normalization                |
| **Conv2D (64 filters, 3x3)** | (16, 16, 64) | ~37K      | ReLU activation, same padding |
| **BatchNormalization**      | (16, 16, 64) | ~256      | Normalization                |
| **MaxPooling2D (2x2)**      | (8, 8, 64)   | 0         | Pooling                      |
| **Dropout (0.25)**          | (8, 8, 64)   | 0         | Regularization               |
| **Conv2D (128 filters, 3x3)**| (8, 8, 128)  | ~73K      | ReLU activation, same padding |
| **BatchNormalization**      | (8, 8, 128)  | ~512      | Normalization                |
| **Conv2D (128 filters, 3x3)**| (8, 8, 128)  | ~147K     | ReLU activation, same padding |
| **BatchNormalization**      | (8, 8, 128)  | ~512      | Normalization                |
| **MaxPooling2D (2x2)**      | (4, 4, 128)  | 0         | Pooling                      |
| **Dropout (0.25)**          | (4, 4, 128)  | 0         | Regularization               |
| **Flatten**                 | (2048)       | 0         | Flatten                      |
| **Dense (256, ReLU)**       | (256)        | ~524K     | Fully connected              |
| **BatchNormalization**      | (256)        | ~1K       | Normalization                |
| **Dropout (0.5)**           | (256)        | 0         | Regularization               |
| **Dense (128, ReLU)**       | (128)        | ~33K      | Fully connected              |
| **BatchNormalization**      | (128)        | ~512      | Normalization                |
| **Dropout (0.5)**           | (128)        | 0         | Regularization               |
| **Dense (10, softmax)**     | (10)         | ~1.3K     | Output layer                 |

- **Optimizer**: Adam (default learning rate). ⚙️  
- **Loss Function**: Sparse Categorical Crossentropy.  
- **Regularization**: BatchNormalization (for faster convergence), Dropout (to prevent overfitting).  
- **Input Shape**: (32, 32, 3).  

The model progressively increases channels (32→64→128) during training and uses MaxPooling to reduce dimensions. 🧠

## 🔬 Experiment Design

### 1. **Environment Setup** 💻
- Python 3.x, GPU support (Colab T4 GPU).  
- Random seeds fixed (42) for reproducibility.  

### 2. **Training Workflow** 🏃‍♂️
- **Batch Size**: 64.  
- **Epochs**: Up to 50 (actual early stopping at best point).  
- **Callbacks**:  
  - **EarlyStopping**: Monitors val_accuracy, patience 10 epochs, restores best weights.  
  - **ReduceLROnPlateau**: Monitors val_loss, reduces LR by 0.5, patience 5 epochs, min LR 1e-7.  
  - **ModelCheckpoint**: Saves best model (val_accuracy).  
- **Data Flow**: Training with augmentation generator, fixed test set.  

### 3. **Evaluation and Analysis** 📊
- **Metrics**: Accuracy, loss, classification report (Precision/Recall/F1).  
- **Visualizations**:  
  - Training history curves (accuracy/loss).  
  - Confusion matrix heatmap.  
  - Per-class accuracy bar chart (with average line).  
- **Saving**: Model saved as `cifar10_cnn_model.h5` and `best_cifar10_model.h5`.  

### 4. **Runtime** ⏱️
- Training time: ~30-60 minutes (depending on GPU).  

## 📈 Experiment Results

### 📊 Performance Summary
| Metric                | Value   |
|-----------------------|---------|
| Final Train Accuracy  | 86.63%  |
| Final Val Accuracy    | 87.39%  |
| Test Accuracy         | 87.70%  |
| Training Epochs       | 50      |
| Best Val Accuracy     | 87.70%  |

### 🎯 Class Analysis
- **Hardest Class**: Cat (accuracy 70.40%) — possibly due to complex fur textures. 😿  
- **Easiest Class**: Frog (accuracy 95.90%) — distinct shape features. 🐸  
- Average class accuracy: ~87.70%.  
<img width="1148" height="559" alt="image" src="https://github.com/user-attachments/assets/31ad2cad-fb49-4975-835d-79a15d7298de" />
*(Bar chart: Green > average, red < average.)* 📊


-Accuracy curves: Training/validation converge smoothly.* 📉
<img width="713" height="459" alt="image" src="https://github.com/user-attachments/assets/2d25bdcb-ed12-4ebd-9f29-4477d1390da8" />



-Heatmap shows main confusions: Cat with Dog/Bird.* 🔍
<img width="801" height="644" alt="image" src="https://github.com/user-attachments/assets/de0f8830-e872-4ce9-b197-0fa4951976a1" />
Strengths (Top Classes, >90% Accuracy):
Frog: 95.9% – Minimal errors due to unique shape.
Automobile: 95.5% – Clear outlines shine.
Ship/Truck: 94.0%/94.8% – Transport categories excel.

Weaknesses (Bottom Classes, <80% Accuracy):
Cat: 70.4% – Frequently confused with Dog (83 cases), Frog (68), Bird (51) from texture/pose overlap.
Dog: 76.5% – 101 misclassified as Cat; 34 as Deer.
Bird: 84.6% – 50 to Frog; 24 to Cat.

### 📋 Classification Report (Summary)
```
              precision    recall  f1-score   support
Airplane         0.90      0.92      0.91      1000
... (see Notebook for details)
accuracy                            0.88     10000
```

## 💻 Installation and Usage

### 📦 Dependencies
```bash
pip install tensorflow matplotlib seaborn scikit-learn pandas numpy
```

### ▶️ Running
1. Clone/download the project.  
2. Open `2_CIFAR_10_Image_Classification_with_Neural_Networks.ipynb`.  
3. Run all cells in Jupyter/Colab (GPU recommended).  
4. Outputs: Model files, charts, result prints.  

### ⚙️ Customization
- Modify augmentation parameters like `rotation_range` for different experiments.  
- Adjust `epochs` or callback patience for robustness testing.  

For issues, welcome to open an Issue! ❗  
**Author**: Generated from Jupyter Notebook experiment.  
**Date**: November 09, 2025
