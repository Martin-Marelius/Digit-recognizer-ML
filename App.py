import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and TensorFlow is using it.")
else:
    print("GPU is not available. TensorFlow is using the CPU.")

train_data = pd.read_csv('MNIST_CSV/mnist_test.csv')
test_data = pd.read_csv('MNIST_CSV/mnist_train.csv')

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

model = Sequential([
    Conv2D(filters=28, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.0005)),
    Conv2D(filters=28, kernel_size=5, strides=1, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Dropout(0.25),
    Conv2D(filters=56, kernel_size=3, strides=1, activation='relu', kernel_regularizer=l2(0.0005)),
    Conv2D(filters=56, kernel_size=3, strides=1, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Dropout(0.25),
    Flatten(),
    Dense(units=280, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(units=140, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(units=70, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

predictions = model.predict(X_test)

for i in range(10):
    actual_digit = np.argmax(y_test[i])
    predicted_digit = np.argmax(predictions[i])
    print(f"Test Image {i + 1}: Actual Digit = {actual_digit}, Predicted Digit = {predicted_digit}")

model.save('mnist_model.h5')
print("Model saved as mnist_model.h5")
