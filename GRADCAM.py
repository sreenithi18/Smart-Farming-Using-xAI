import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import json

# Load the pre-trained model
model = load_model('paddy-doctor-final-cnn.keras')

# Path to the test image
test_image_path = 'dataset/test/bacterial_leaf_streak/PDD00876_003.jpg'

# Preprocess image for the model
def preprocess_image_for_model(img_path, target_size=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale as per your training setup
    return img_array

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    conv_outputs *= pooled_grads

    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Function to overlay heatmap on image
def overlay_gradcam(heatmap, img_path, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

# Function to save the image instead of displaying
def save_img(img, output_path='gradcam_result.jpg'):
    cv2.imwrite(output_path, img)
    print(f"Grad-CAM result saved to {output_path}")

# Function to evaluate model and save metrics
def evaluate_model(model, img_array, true_label=None):
    # Predict on the input image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    prediction_confidence = np.max(predictions)
    
    # Calculate metrics (for this example, only predicted class and confidence are stored)
    metrics = {
        "predicted_class": int(predicted_class),
        "prediction_confidence": float(prediction_confidence)
    }
    
    # If you have true labels for the image, you can also add accuracy
    if true_label is not None:
        accuracy = 1 if predicted_class == true_label else 0
        metrics["accuracy"] = accuracy
    
    # Save metrics to a JSON file
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to metrics.json")
    return metrics

# Function to calculate XAI metrics
def print_xai_metrics(model, img_array, heatmap):
    # Faithfulness metric
    faithfulness = compute_faithfulness(model, img_array, heatmap)
    print(f"Faithfulness: {faithfulness:.4f}")
    
    # Localization Accuracy (dummy example, adjust based on ground truth localization info)
    localization_accuracy = np.mean(heatmap > 0.5)  # Assuming higher activations indicate better localization
    print(f"Localization Accuracy: {localization_accuracy:.4f}")
    
    # Robustness: Slight perturbation to the input image
    robustness = compute_robustness(model, img_array, heatmap)
    print(f"Robustness: {robustness:.4f}")
    
    # Class Discriminativity: How well the model differentiates classes
    class_discriminativity = compute_class_discriminativity(model, img_array)
    print(f"Class Discriminativity: {class_discriminativity:.4f}")
    
    # Sparsity: Fraction of important regions in heatmap
    sparsity = np.sum(heatmap < 0.1) / heatmap.size
    print(f"Sparsity: {sparsity:.4f}")
    
    # Completeness: Fraction of predicted region covered by heatmap
    completeness = np.sum(heatmap)  # Placeholder calculation
    print(f"Completeness: {completeness:.4f}")
    
    # Sensitivity: Change in model output with respect to perturbations
    sensitivity = compute_sensitivity(model, img_array, heatmap)
    print(f"Sensitivity: {sensitivity:.4f}")

# Additional XAI metrics functions (example placeholders)

def compute_faithfulness(model, img_array, heatmap):
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    threshold = np.percentile(heatmap_resized, 90)
    masked_img = img_array.copy()
    masked_img[0, :, :, :][heatmap_resized < threshold] = 0  # Set low-importance areas to zero
    orig_pred = np.max(model.predict(img_array))
    masked_pred = np.max(model.predict(masked_img))
    faithfulness = orig_pred - masked_pred
    return faithfulness

def compute_robustness(model, img_array, heatmap):
    noise = np.random.normal(0, 0.05, img_array.shape)
    noisy_img = img_array + noise
    noisy_pred = np.max(model.predict(noisy_img))
    orig_pred = np.max(model.predict(img_array))
    return np.abs(orig_pred - noisy_pred)

def compute_class_discriminativity(model, img_array):
    orig_pred = np.max(model.predict(img_array))
    shuffled_img = np.random.permutation(img_array.flatten()).reshape(img_array.shape)
    shuffled_pred = np.max(model.predict(shuffled_img))
    return orig_pred - shuffled_pred

def compute_sensitivity(model, img_array, heatmap):
    perturbed_img = img_array + np.random.uniform(-0.01, 0.01, img_array.shape)
    perturbed_pred = np.max(model.predict(perturbed_img))
    orig_pred = np.max(model.predict(img_array))
    return np.abs(orig_pred - perturbed_pred)

# Main Grad-CAM process
last_conv_layer_name = 'conv2d_2'  # Updated based on your model summary
img_array = preprocess_image_for_model(test_image_path)

# Generate Grad-CAM heatmap
heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)

# Overlay the heatmap on the original image
superimposed_img = overlay_gradcam(heatmap, test_image_path)

# Save the Grad-CAM result as an image file
save_img(superimposed_img, 'gradcam_result.jpg')

# Evaluate the model and save metrics
metrics = evaluate_model(model, img_array, true_label=None)  # Add true_label if available

# Print XAI metrics
print_xai_metrics(model, img_array, heatmap)

