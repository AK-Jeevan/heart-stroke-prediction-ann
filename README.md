# 🧠 Stroke Prediction using ANN

This project uses an Artificial Neural Network (ANN) to classify whether a patient has experienced a stroke based on key health indicators.

---

## 🗂️ Dataset Overview

- **Source**: [Healthcare Dataset - Stroke Data on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Features**:
  - `age`: Patient age
  - `hypertension`: 0 or 1
  - `heart_disease`: 0 or 1
  - `avg_glucose_level`: Average glucose level in blood
  - `bmi`: Body Mass Index
  - `smoking_status`: Encoded as categorical integers
- **Target**: `stroke` (1 if patient had a stroke, else 0)

---

## 🧠 Model Architecture

A deep ANN built with Keras and TensorFlow:

- Input: 6 health features
- Hidden Layers:
  - Dense(64) → BatchNormalization → Dropout(0.4)
  - Dense(32) → BatchNormalization → Dropout(0.3)
  - Dense(16)
- Output: Dense(1) with sigmoid activation
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Regularization: Dropout + EarlyStopping

---

## 🔄 Workflow

1. **Data Cleaning**: Remove duplicates and missing values
2. **Feature Encoding**: Label encode `smoking_status`
3. **Train-Test Split**: 75/25 with stratification
4. **Scaling**: StandardScaler for normalization
5. **Model Training**: 150 epochs with validation split and early stopping
6. **Evaluation**: Accuracy score and confusion matrix
7. **Visualization**: Training history plotted for performance tracking

---

## 📊 Results

- **Test Accuracy**: ~`[Insert your result here]`
- **Confusion Matrix**:
- **Training History**: Accuracy vs. Epochs plotted using Matplotlib

---

## 🚀 How to Run

### 🧰 Requirements
pip install numpy pandas matplotlib scikit-learn tensorflow keras

##📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code with proper attribution.
