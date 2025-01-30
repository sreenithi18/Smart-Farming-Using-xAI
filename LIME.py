import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
import cv2

# Use the same image processing steps as in your training
def preprocess_image_for_model(img_path, target_size=(256, 256)):
    # Load image and resize to the target size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale as per your training setup
    return img_array

# Function for LIME explanation
def lime_explanation(model, img_path, num_samples=1000):
    # Preprocess image for the model
    img_array = preprocess_image_for_model(img_path)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation for the image
    explanation = explainer.explain_instance(
        img_array[0].astype('double'),  # Use the preprocessed image
        model.predict,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )

    # Get prediction for the top class
    top_label = np.argmax(model.predict(img_array))

    # Return explanation and mask for the top class
    return explanation, top_label

# Display LIME explanation
def display_lime_explanation(explanation, top_label,save_path='lime_explanation.png'):
    temp, mask = explanation.get_image_and_mask(
        top_label, 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )

    # Display explanation with boundaries
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    

# Load your pre-trained model (ensure it matches the model saved after training)
model = load_model('paddy-doctor-final-cnn.keras')

# Path to the test image
test_image_path = 'dataset/test/bacterial_leaf_streak/PDD00876_003.jpg'

# Generate explanation using LIME
explanation, top_label = lime_explanation(model, test_image_path)
display_lime_explanation(explanation, top_label)

# Define new metrics

# 1. Fidelity (Modified)
def fidelity(model, lime_explanation, img_array, original_prediction, num_samples=1000):
    segments = lime_explanation.segments
    perturbed_images = []

    # Generate perturbed images by setting segments to the mean value
    for seg_val in np.unique(segments):
        perturbed_image = img_array[0].copy()
        mask = segments == seg_val
        perturbed_image[mask] = np.mean(perturbed_image[mask], axis=0)
        perturbed_images.append(perturbed_image)
    
    perturbed_images = np.array(perturbed_images)
    # Predict for all perturbed images
    perturbed_preds = model.predict(perturbed_images)
    # Calculate the fidelity based on how many predictions match the original prediction
    matching_labels = np.sum(np.argmax(perturbed_preds, axis=1) == original_prediction)
    
    return matching_labels / len(perturbed_images)  # Fidelity score

# 2. Sparsity
def sparsity(lime_explanation, top_label):
    return len(lime_explanation.local_exp[top_label])

# 3. Faithfulness
def faithfulness(model, original_image, explanation, top_label):
    # Get the mask for the top predicted label
    _, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand mask to 3D shape
    # Apply the mask to the original image
    modified_image = original_image * mask_3d
    # Predict the model's output for the modified image
    modified_prediction = model.predict(np.expand_dims(modified_image, axis=0))
    # Calculate faithfulness as the drop in prediction score for the top label
    original_prediction = model.predict(np.expand_dims(original_image, axis=0))
    return original_prediction[0][top_label] - modified_prediction[0][top_label]

# 4. Monotonicity
def monotonicity(model, original_image, explanation, top_label):
    # Get the mask for the top predicted label
    _, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Apply the mask to the original image
    modified_image = original_image * mask_3d
    # Predict the model's output for the modified image
    modified_prediction = model.predict(np.expand_dims(modified_image, axis=0))
    # Calculate monotonicity score
    original_prediction = model.predict(np.expand_dims(original_image, axis=0))
    return original_prediction[0][top_label] - modified_prediction[0][top_label]

# 5. Completeness
def completeness(model, original_image, lime_explanation, top_label, top_k=5):
    temp, mask = lime_explanation.get_image_and_mask(
        top_label, 
        positive_only=True, 
        num_features=top_k, 
        hide_rest=True
    )
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    modified_image = original_image * mask_3d
    modified_prediction = np.argmax(model.predict(np.expand_dims(modified_image, axis=0)))
    return modified_prediction == top_label

# 6. Robustness
def robustness(model, lime_explanation, original_image, top_label, epsilon=0.01):
    noise = np.random.normal(0, epsilon, original_image.shape)
    noisy_image = original_image + noise
    new_explanation = lime_image.LimeImageExplainer().explain_instance(
        noisy_image.astype('double'),
        model.predict,
        top_labels=5,  # Specify the number of top labels instead of passing a list
        hide_color=0,
        num_samples=1000
    )
    original_features = set([f[0] for f in lime_explanation.local_exp[lime_explanation.top_labels[0]]])
    new_features = set([f[0] for f in new_explanation.local_exp[new_explanation.top_labels[0]]])
    return len(original_features.intersection(new_features)) / len(original_features)


# Evaluate LIME explanation using new metrics
def evaluate_metrics(model, explanation, top_label, img_array):
    original_prediction = np.argmax(model.predict(img_array))
    print(f"Fidelity: {fidelity(model, explanation, img_array, original_prediction):.4f}")
    print(f"Sparsity: {sparsity(explanation, top_label)} features")
    print(f"Faithfulness: {faithfulness(model, img_array[0], explanation, top_label):.4f}")
    print(f"Monotonicity: {monotonicity(model, img_array[0], explanation, top_label):.4f}")
    print(f"Completeness: {completeness(model, img_array[0], explanation, top_label)}")
    print(f"Robustness: {robustness(model, explanation, img_array[0], top_label):.4f}")

# Optional: Evaluate the explanation using new metrics
evaluate_metrics(model, explanation, top_label, preprocess_image_for_model(test_image_path))

