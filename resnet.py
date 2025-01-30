# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Define image dimensions
IMAGE_SIZE = [256, 256]  # Adjust based on your dataset

# Define paths to your dataset
train_path = '/home/psg/dataset/train'
valid_path = '/home/psg/dataset/test'

# Preprocessing (if the dataset is already augmented, just rescale)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset
train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

valid_set = valid_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Load the ResNet50 model with pre-trained weights
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the ResNet50 layers to retain pre-trained weights
for layer in resnet.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
x = Flatten()(resnet.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(train_set.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=resnet.input, outputs=output)

# Compile the model with additional metrics
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']  # Accuracy will be reported, others will be computed post-evaluation
)

# Set up model checkpoints and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=100,  # Adjust based on your requirement
    steps_per_epoch=len(train_set),
    validation_steps=len(valid_set),
    callbacks=[checkpoint, early_stop]
)

# Save the final model
model.save('resnet50_crop_disease_detection.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(valid_set)

# Print the accuracy score
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Predict on the validation set to generate a classification report and confusion matrix
y_pred = model.predict(valid_set)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_set.classes

# Generate the classification report
print(classification_report(y_true, y_pred_classes, target_names=valid_set.class_indices.keys()))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# Optionally, plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_validation_accuracy.png')  # Save the plot to a file

plt.figure()  # Create a new figure
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss.png')  # Save the plot to a file



