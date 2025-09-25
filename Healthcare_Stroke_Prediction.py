# This Model is used to predict whether a patient had a Heart Stroke or not using Artificial Neural Networks (ANN)
# Classification Problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Step 1: Load and inspect the data
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\DL\ANN\healthcare-dataset-stroke-data.csv")
print(data.head())
print(data.describe())

# Step 2: Clean the data
data = data.drop_duplicates()
data = data.dropna().reset_index(drop=True)

# Encode 'smoking_status' as numbers (label encoding, each type gets a unique integer)
data['smoking_status'] = data['smoking_status'].astype('category').cat.codes

# Step 3: Separate features and target variable
x = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi','smoking_status']]
y = data['stroke']

# Step 4: Split the data into training and testing sets (stratify for balanced classes)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Build the ANN model (deeper, with batch normalization)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Step 7: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1, mode='min', min_delta=0.0005
)

# Step 9: Train the model
history = model.fit(
    x_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.25,
    callbacks=[early_stopping],
    verbose=1
)

# Step 10: Evaluate the model
y_pred_prob = model.predict(x_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc:.3f}")
print("Confusion Matrix:")
print(cm)

# Step 11: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Training History')
plt.show()