import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configured successfully.")
    except Exception as e:
        print("Failed to configure GPU:", e)

# Set random seeds for reproducibility
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)

# MRDOA Parameters
POPULATION_SIZE = 10  # Number of deers (hyperparameter sets)
GENERATIONS = 10      # Number of generations
MATING_PROB = 0.7     # Probability of mating
MUTATION_RATE = 0.1   # Probability of mutation
MATING_FACTOR = 0.5   # Controls offspring generation


# Search space for hyperparameters
search_space = {
    "learning_rate": [1e-4, 1e-5, 5e-5],  # Narrow down the learning rates
    "batch_size": [16, 32],  # Focus on smaller batch sizes
    "dropout_rate": [0.2, 0.3]  # Lower dropout rates for more stability
}


# Sample initial population
def initialize_population(pop_size, search_space):
    population = []
    for _ in range(pop_size):
        deer = {
            "learning_rate": random.choice(search_space["learning_rate"]),
            "batch_size": random.choice(search_space["batch_size"]),
            "dropout_rate": random.choice(search_space["dropout_rate"])
        }
        population.append(deer)
    return population

# Evaluate fitness (validation accuracy) of the model based on hyperparameters
def evaluate_fitness(deer, model, train_generator, validation_generator):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=deer['learning_rate']), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    history = model.fit(train_generator, 
                        epochs=5, 
                        batch_size=deer['batch_size'], 
                        validation_data=validation_generator, 
                        verbose=0)
    
    validation_accuracy = history.history['val_accuracy'][-1]
    return validation_accuracy

# Perform mating to generate offspring
def mate(parent1, parent2):
    child = {}
    for param in search_space.keys():
        if random.random() < MATING_FACTOR:
            child[param] = parent1[param]
        else:
            child[param] = parent2[param]
    
    # Mutation step
    if random.random() < MUTATION_RATE:
        param_to_mutate = random.choice(list(search_space.keys()))
        child[param_to_mutate] = random.choice(search_space[param_to_mutate])
    
    return child

# Select top males for reproduction based on fitness
def select_males(population, fitness_scores, num_males):
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort by fitness in descending order
    top_males = [population[i] for i in sorted_indices[:num_males]]
    return top_males

# MRDOA Optimization Loop
def mrdoa_optimize(model, train_generator, validation_generator, generations, pop_size):
    population = initialize_population(pop_size, search_space)
    best_deer = None
    best_fitness = -np.inf
    
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness for the current population
        fitness_scores = []
        for deer in population:
            fitness = evaluate_fitness(deer, model, train_generator, validation_generator)
            fitness_scores.append(fitness)
        
        # Select the best male deer based on fitness
        top_males = select_males(population, fitness_scores, pop_size // 2)
        
        # Create new offspring by mating
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(top_males, 2)
            child = mate(parent1, parent2)
            new_population.append(child)
        
        # Combine parents and offspring to form new population
        population = top_males + new_population
        
        # Track the best deer (best solution)
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_deer = population[np.argmax(fitness_scores)]
        
        print(f"Best Fitness in Generation {generation + 1}: {best_fitness}")
    
    print("Optimization Complete!")
    print(f"Best Deer Hyperparameters: {best_deer}")
    print(f"Best Fitness Achieved: {best_fitness}")
    
    return best_deer

# Create the CNN model
def create_model(input_shape, num_classes, dropout_rate):
    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the ResNet50 layers to not update weights during training
    base_model.trainable = False  
    
    # Add custom layers on top of ResNet50
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Use Global Average Pooling layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)  # Use dropout to prevent overfitting
    
    # Final output layer with softmax for multi-class classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model

# Set image dimensions and number of classes
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 13  # Example number of classes for paddy diseases
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# Data generators for image augmentation and loading
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)  # 20% for validation
train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Path to the training dataset folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # Set as training data
    seed=SEED
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Use the same training directory for validation
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Set as validation data
    seed=SEED
)

# Create the initial model
model = create_model(input_shape=input_shape, num_classes=NUM_CLASSES, dropout_rate=0.5)

# Run MRDOA optimization to tune hyperparameters
best_deer = mrdoa_optimize(model, train_generator, validation_generator, generations=GENERATIONS, pop_size=POPULATION_SIZE)

# Final model with optimized hyperparameters
final_model = create_model(input_shape=input_shape, num_classes=NUM_CLASSES, dropout_rate=best_deer['dropout_rate'])
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_deer['learning_rate']), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

# Train the model with the best hyperparameters
final_model.fit(train_generator, 
                epochs=1000,  # Set the number of epochs you want
                batch_size=best_deer['batch_size'], 
                validation_data=validation_generator)

# Testing the model on the separate test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for test data
test_generator = test_datagen.flow_from_directory(
    'dataset/test',  # Path to the test dataset folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do not shuffle for evaluation
)

# Evaluate the final model on the test dataset
test_loss, test_accuracy = final_model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

