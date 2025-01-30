import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Function to load a single image
def load_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to apply occlusion sensitivity
def occlusion_sensitivity(model, image, patch_size=16, stride=8):
    original_prediction = model.predict(image)[0]
    sensitivity_map = np.zeros((image.shape[1], image.shape[2]))  # Same spatial size as input image
    img_height, img_width = image.shape[1], image.shape[2]
    
    for h in range(0, img_height, stride):
        for w in range(0, img_width, stride):
            # Create a copy of the image
            occluded_image = image.copy()
            
            # Apply occlusion (e.g., zero out a patch in the image)
            occluded_image[:, h:h+patch_size, w:w+patch_size, :] = 0
            
            # Get model's prediction on the occluded image
            occluded_prediction = model.predict(occluded_image)[0]
            
            # Compute the change in the model's confidence for the true class
            confidence_change = original_prediction - occluded_prediction
            
            # Aggregate the confidence change into the sensitivity map
            sensitivity_map[h:h+patch_size, w:w+patch_size] = np.mean(confidence_change)
    
    return sensitivity_map

# Function to plot the original image and occlusion sensitivity map
def plot_occlusion_sensitivity(original_image, sensitivity_map, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Sensitivity map
    im = ax[1].imshow(sensitivity_map, cmap='hot', interpolation='nearest')
    ax[1].set_title('Occlusion Sensitivity Map')
    ax[1].axis('off')
    
    # Add color bar to show the scale
    fig.colorbar(im, ax=ax[1])

    if save_path:
        plt.savefig(save_path)
        print(f"Sensitivity map saved as {save_path}")
    plt.show()

# Load your trained model
model = tf.keras.models.load_model('paddy-doctor-best-cnn.keras')

# Load a test image (adjust the path accordingly)
image_path = 'dataset/test/hispa/PDD07176_001.jpg'  # Replace with your actual test image
test_image = load_image(image_path)

# Compute the occlusion sensitivity map
sensitivity_map = occlusion_sensitivity(model, test_image)

# Plot and save the occlusion sensitivity map
plot_occlusion_sensitivity(test_image[0], sensitivity_map, save_path='occlusion_sensitivity1.png')

