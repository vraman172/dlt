# Step 1: Import Required Libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 2: Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 3: Preprocess the Data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Reshape & normalize
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Step 4 & 5: Initialize and Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional Layer
    MaxPooling2D((2, 2)),                                            # Pooling Layer
    Flatten(),                                                       # Flatten Layer
    Dense(128, activation='relu'),                                   # Dense Hidden Layer
    Dense(10, activation='softmax')                                  # Output Layer
])

# Step 6: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train the Model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
