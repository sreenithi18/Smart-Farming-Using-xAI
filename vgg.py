# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

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

# Load the VGG16 model with pre-trained weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the VGG16 layers to retain pre-trained weights
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers on top of VGG16
x = GlobalAveragePooling2D()(vgg.output)  # Use Global Average Pooling
x = Dense(1024, activation='relu')(x)
output = Dense(train_set.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=vgg.input, outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Set up model checkpoints and early stopping
checkpoint = ModelCheckpoint('best_model_vgg.keras', monitor='val_accuracy', save_best_only=True, mode='max')
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
model.save('vgg16_crop_disease_detection.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(valid_set)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Generate predictions on validation data
val_predictions = model.predict(valid_set)
val_pred_classes = np.argmax(val_predictions, axis=1)  # Get predicted class indices
true_classes = valid_set.classes  # Get true class indices

# Generate classification report
class_labels = list(valid_set.class_indices.keys())  # Get class labels from the dataset
report = classification_report(true_classes, val_pred_classes, target_names=class_labels)
print("Classification Report:\n", report)

# Optionally, print confusion matrix
conf_matrix = confusion_matrix(true_classes, val_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Calculate Precision, Recall, and F1-score
precision = precision_score(true_classes, val_pred_classes, average='weighted')
recall = recall_score(true_classes, val_pred_classes, average='weighted')
f1 = f1_score(true_classes, val_pred_classes, average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")

# Optionally, plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_validation_accuracy_vgg.png')  # Save the plot to a file

plt.figure()  # Create a new figure
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss_vgg.png')  # Save the plot to a file

