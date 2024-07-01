import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# Шлях до файлу mnist.npz
path = '/Users/diana/Desktop/Нейрообчислення/mnist.npz'

# Завантаження локального файлу MNIST
with np.load(path, allow_pickle=True) as f:
    train_images, train_labels = f['x_train'], f['y_train']
    test_images, test_labels = f['x_test'], f['y_test']

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Збільшення розміру зображень до 32x32 для VGG16
train_images_resized = np.array([tf.image.resize(img, [32, 32]) for img in train_images])
test_images_resized = np.array([tf.image.resize(img, [32, 32]) for img in test_images])

# Розбиття тренувального набору на тренувальний та валідаційний
val_images = train_images[:10000]
val_labels = train_labels[:10000]
train_images = train_images[10000:]
train_labels = train_labels[10000:]

val_images_resized = train_images_resized[:10000]
train_images_resized = train_images_resized[10000:]

# Функція для створення та тренування моделі з логуванням ресурсів
def train_model_with_resources(model, train_images, train_labels, val_images, val_labels, epochs=10):
    start_time = time.time()
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))
    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")
    
    return history, training_time

# Модель багатошарового перцептрону (MLP)
mlp_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')  # 10 класів
])
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Проста CNN модель
simple_cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 класів
])
simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Передтренована модель VGG16
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16_base.trainable = False
advanced_model = Sequential([
    Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu'),  # Додатковий шар для приведення до 3 каналів
    vgg16_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # 10 класів
])
advanced_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделей
mlp_info = train_model_with_resources(mlp_model, train_images, train_labels, val_images, val_labels, epochs=10)
simple_cnn_info = train_model_with_resources(simple_cnn_model, train_images, train_labels, val_images, val_labels, epochs=10)
advanced_info = train_model_with_resources(advanced_model, train_images_resized, train_labels, val_images_resized, val_labels, epochs=10)

# Функція для порівняльного візуалізування результатів тренування
def compare_models(histories, model_names):
    colors = ['b', 'r', 'g']
    plt.figure(figsize=(14, 5))
    for i, history in enumerate(histories):
        plt.subplot(1, 2, 1)
        plt.plot(history[0].history['accuracy'], color=colors[i], label=f'{model_names[i]} Training Accuracy')
        plt.plot(history[0].history['val_accuracy'], color=colors[i], linestyle='dashed', label=f'{model_names[i]} Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(history[0].history['loss'], color=colors[i], label=f'{model_names[i]} Training Loss')
        plt.plot(history[0].history['val_loss'], color=colors[i], linestyle='dashed', label=f'{model_names[i]} Validation Loss')
        
    plt.subplot(1, 2, 1)
    plt.title('Model Accuracy Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Model Loss Comparison')
    plt.legend()

    plt.show()

# Візуалізація порівнянь
compare_models([mlp_info, simple_cnn_info, advanced_info], ['MLP', 'Simple CNN', 'Advanced VGG16'])
