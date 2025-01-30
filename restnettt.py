# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def conv_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    """A block that has a conv layer at shortcut."""
    shortcut = x
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride)(x)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def identity_block(x, filters, kernel_size=3):
    """A block that has no conv layer at shortcut."""
    shortcut = x

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def resnet34(input_shape=(256, 256, 3), num_classes=10):
    """Build ResNet34 architecture."""
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1
    x = conv_block(x, 64, stride=1)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    # Stage 2
    x = conv_block(x, 128, stride=2)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    # Stage 3
    x = conv_block(x, 256, stride=2)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)

    # Stage 4
    x = conv_block(x, 512, stride=2)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    # Final layers
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Load the ResNet34 model
resnet = resnet34(num_classes=train_set.num_classes)

# Compile the model
resnet.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Set up model checkpoints and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = resnet.fit(
    train_set,
    validation_data=valid_set,
    epochs=35,  # Adjust based on your requirement
    steps_per_epoch=len(train_set),
    validation_steps=len(valid_set),
    callbacks=[checkpoint]
)

# Save the final model
resnet.save('resnet34_crop_disease_detection.h5')

# Evaluate the model on the validation set
loss, accuracy = resnet.evaluate(valid_set)

# Print the accuracy score
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Optionally, plot accuracy and loss
import matplotlib.pyplot as plt

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

