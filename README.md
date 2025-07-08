"# 🥔 Potato Skin Disease Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🔬 An AI-powered computer vision system for detecting and classifying potato skin diseases using deep learning techniques.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🌟 Features](#-features)
- [📊 Dataset](#-dataset)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [🏗️ Model Architecture](#️-model-architecture)
- [📈 Results](#-results)
- [🔧 Installation](#-installation)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to automatically detect and classify potato skin diseases from digital images. The system can identify three main categories:

- 🍃 **Healthy Potatoes**
- 🦠 **Early Blight Disease**
- 🍄 **Late Blight Disease**

### 🎥 Demo

<details>
<summary>Click to see sample predictions</summary>

```
Input: potato_image.jpg
Output: "Early Blight Disease" (Confidence: 94.2%)
```

</details>

## 🌟 Features

- ✅ **Multi-class Classification**: Detects 3 types of potato conditions
- ✅ **Data Augmentation**: Improves model robustness with image transformations
- ✅ **Interactive Visualization**: Displays sample images with predictions
- ✅ **Optimized Performance**: Uses caching and prefetching for faster training
- ✅ **Scalable Architecture**: Easy to extend to more disease types
- ✅ **Real-time Inference**: Fast prediction on new images

## 📊 Dataset

### 📁 Dataset Structure

```
PlantVillage/
├── Potato___Early_blight/     # 🦠 Early blight disease images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Potato___Late_blight/      # 🍄 Late blight disease images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Potato___healthy/          # 🍃 Healthy potato images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 📈 Dataset Statistics

- **Total Images**: 2,152
- **Classes**: 3 (Early Blight, Late Blight, Healthy)
- **Image Size**: 256×256 pixels
- **Color Channels**: RGB (3 channels)
- **Data Split**: 80% Train, 10% Validation, 10% Test

## 🚀 Getting Started

### 📋 Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Matplotlib
NumPy
```

### ⚡ Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/potato-disease-detection.git
   cd potato-disease-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook POTATO_Skin_Diseases_Detection_Using_Deep_Learning.ipynb
   ```

## 💻 Usage

### 🔧 Training the Model

The notebook includes the complete pipeline:

1. **Data Loading & Preprocessing**

   ```python
   # Load dataset
   dataset = tf.keras.preprocessing.image_dataset_from_directory(
       "PlantVillage",
       image_size=(256, 256),
       batch_size=32
   )
   ```

2. **Data Augmentation**

   ```python
   # Apply data augmentation
   data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal_and_vertical"),
       tf.keras.layers.RandomRotation(0.2)
   ])
   ```

3. **Model Configuration**
   ```python
   IMAGE_SIZE = 256
   BATCH_SIZE = 32
   CHANNELS = 3
   EPOCHS = 50
   ```

### 🎯 Making Predictions

```python
# Load your trained model
model = tf.keras.models.load_model('potato_disease_model.h5')

# Make prediction
prediction = model.predict(new_image)
predicted_class = class_names[np.argmax(prediction)]
```

## 🏗️ Model Architecture

### 🧠 Network Components

1. **Input Layer**: 256×256×3 RGB images
2. **Preprocessing**:
   - Image resizing and rescaling (1.0/255)
   - Data augmentation (RandomFlip, RandomRotation)
3. **Feature Extraction**: CNN layers for pattern recognition
4. **Classification**: Dense layers for final prediction

### ⚙️ Training Configuration

- **Optimizer**: Adam (recommended)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32

## 📈 Results

### 📊 Performance Metrics

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | XX.X% |
| Validation Accuracy | XX.X% |
| Test Accuracy       | XX.X% |
| F1-Score            | XX.X% |

### 🎨 Visualization

The notebook includes:

- ✅ Sample image visualization
- ✅ Training/validation loss curves
- ✅ Confusion matrix
- ✅ Class-wise accuracy

## 🔧 Installation

### 🐍 Environment Setup

```bash
# Create virtual environment
python -m venv potato_env

# Activate environment
# Windows:
potato_env\Scripts\activate
# macOS/Linux:
source potato_env/bin/activate

# Install packages
pip install -r requirements.txt
```

## 📁 Project Structure

```
potato-disease-detection/
├── 📓 POTATO_Skin_Diseases_Detection_Using_Deep_Learning.ipynb
├── 📄 README.md
├── 📋 requirements.txt
├── 📁 PlantVillage/
│   ├── 📁 Potato___Early_blight/
│   ├── 📁 Potato___Late_blight/
│   └── 📁 Potato___healthy/
├── 📁 models/
│   └── 💾 trained_model.h5
└── 📁 results/
    ├── 📊 training_plots.png
    └── 📈 confusion_matrix.png
```

## 🚀 Next Steps

### 🔮 Future Enhancements

- [ ] **Model Optimization**: Implement transfer learning with pre-trained models
- [ ] **Web Application**: Create a Flask/Streamlit web interface
- [ ] **Mobile App**: Develop a mobile application for field use
- [ ] **More Diseases**: Expand to detect additional potato diseases
- [ ] **Real-time Detection**: Implement live camera feed processing
- [ ] **API Development**: Create REST API for integration

### 🎯 Improvement Ideas

- [ ] **Hyperparameter Tuning**: Optimize model parameters
- [ ] **Cross-validation**: Implement k-fold cross-validation
- [ ] **Ensemble Methods**: Combine multiple models
- [ ] **Data Balancing**: Handle class imbalance if present

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 Bug Reports

If you find a bug, please create an issue with:

- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information

### 💡 Feature Requests

For new features, please provide:

- Clear description of the feature
- Use case and benefits
- Implementation suggestions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the potato disease dataset
- **TensorFlow Team**: For the amazing deep learning framework
- **Open Source Community**: For inspiration and resources

## 📞 Contact

- **Author**: Lucky Sharma
- **Email**: panditluckysharma42646@gmail.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

---

<div align="center">
  <p>⭐ Star this repository if you found it helpful!</p>
  <p>🍀 Happy coding and may your potatoes be healthy!</p>
</div>
"
