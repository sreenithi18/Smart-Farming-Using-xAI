import glob
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Ensure reproducibility
SEED = 1234
def set_seed(seed=SEED):
    np.random.seed(seed) 
    tf.random.set_seed(seed) 
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("Failed to configure GPU.")

# Paths to datasets
train_path = 'dataset/train'
test_path  = 'dataset/test'

# Load data using ImageDataGenerator
image_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=5,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = image_datagen.flow_from_directory(directory=train_path,
                                                    subset='training',
                                                    target_size=(256, 256),
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,
                                                    seed=SEED)

valid_generator = image_datagen.flow_from_directory(directory=train_path,
                                                    subset='validation',
                                                    target_size=(256, 256),
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,
                                                    seed=SEED)

# Model architecture
def get_model():
    model = Sequential()
    inputShape = (256, 256, 3)
    chanDim = -1

    if backend.image_data_format() == "channels_first":
        inputShape = (3, 256, 256)
        chanDim = 1

    # Layer 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(3, 3)))  # Changed to AveragePooling2D
    model.add(Dropout(0.25))

    # Layer 2
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(2, 2)))  # Changed to AveragePooling2D
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(2, 2)))  # Changed to AveragePooling2D
    model.add(Dropout(0.25))

    # Fully connected layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(train_generator.num_classes))  # Dynamic output based on number of classes
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

model = get_model()
plot_model(model, 'cnn-model.png', show_shapes=True)

# ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath='paddy-doctor-best-cnn.keras',
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min',
                             verbose=1)

# EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

# Training the model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[checkpoint, early_stop],
                    epochs=50,
                    verbose=1)

# Plotting training/validation accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# Testing the model
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_path,
                                                                        target_size=(256, 256),
                                                                        color_mode="rgb",
                                                                        batch_size=1,
                                                                        class_mode=None,
                                                                        shuffle=False)

model.load_weights('paddy-doctor-best-cnn.keras')

predictions = model.predict(test_generator)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Evaluate performance
acc = accuracy_score(true_classes, pred_classes)
print(f"Model accuracy: {acc * 100:.2f}%")

print("Classification Report:")
cls_report = classification_report(true_classes, pred_classes, target_names=class_labels, digits=5)
print(cls_report)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(true_classes, pred_classes, class_labels)

