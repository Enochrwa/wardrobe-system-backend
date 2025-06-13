# This module provides functions for extracting image embeddings using lightweight models.
# It uses MobileNetV2 from TensorFlow Hub for efficient embedding generation.

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from typing import List, Optional, Union
import logging # Added for logging

# Global variable to hold the loaded model
mobilenet_v2_model = None
MODEL_LOADED = False
MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

logger = logging.getLogger(__name__) # Added logger

def _load_model():
    """Loads the MobileNetV2 model from TensorFlow Hub."""
    global mobilenet_v2_model, MODEL_LOADED
    if not MODEL_LOADED:
        try:
            logger.info(f"Loading MobileNetV2 model from {MODEL_URL}...")
            # Load the model from TF Hub
            # Using input_shape ensures the model expects a fixed size, which is good for consistency.
            # MobileNetV2 expects images of size 224x224.
            mobilenet_v2_model = hub.KerasLayer(MODEL_URL, input_shape=(224, 224, 3))
            MODEL_LOADED = True
            logger.info("MobileNetV2 model loaded successfully.")
        except Exception as e:
            mobilenet_v2_model = None
            MODEL_LOADED = False
            logger.error(f"Error loading MobileNetV2 model: {e}")
            # Depending on policy, could raise here or allow fallback in extract function

def get_image_embedding(image: Image.Image) -> Union[List[float], str]:
    """
    Extracts image embedding using MobileNetV2.

    Args:
        image: A PIL Image object.

    Returns:
        A list of floats representing the image embedding, or a string with an error message.
    """
    global mobilenet_v2_model, MODEL_LOADED

    if not MODEL_LOADED and mobilenet_v2_model is None:
        _load_model() # Attempt to load the model if not already loaded

    if not mobilenet_v2_model:
        return "Error: Image embedding model (MobileNetV2) could not be loaded. Cannot extract embedding."

    try:
        # Preprocess the image
        # 1. Convert to RGB if it's not
        image = image.convert("RGB")
        # 2. Resize to MobileNetV2's expected input size (224x224)
        image_resized = image.resize((224, 224))
        # 3. Convert to numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image_resized) / 255.0
        # 4. Add a batch dimension (model expects batch_size, height, width, channels)
        image_batch = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Generate embedding
        embedding_tensor = mobilenet_v2_model(image_batch)

        # The output of hub.KerasLayer is already a tensor, often (1, num_features)
        # We convert it to a list of floats
        embedding = embedding_tensor.numpy().flatten().tolist()

        return embedding
    except Exception as e:
        logger.error(f"Error during image embedding extraction with MobileNetV2: {e}")
        return f"Error during image embedding extraction: {str(e)}"

# Example usage (optional, can be removed or commented out)
if __name__ == '__main__':
    # This part is for testing the module directly.
    # It requires Pillow and TensorFlow to be installed.
    # Create a dummy black image for testing
    try:
        img = Image.new('RGB', (224, 224), color = 'black')
        logger.info("Attempting to extract embedding for a dummy image...")
        # Ensure model is loaded for the test
        if not MODEL_LOADED and mobilenet_v2_model is None:
             _load_model()

        if mobilenet_v2_model:
            embedding_result = get_image_embedding(img)
            if isinstance(embedding_result, list):
                logger.info(f"Successfully extracted embedding (first 10 features): {embedding_result[:10]}")
                logger.info(f"Embedding dimension: {len(embedding_result)}")
            else:
                logger.error(f"Embedding extraction failed: {embedding_result}")
        else:
            logger.warning("Skipping example usage as model failed to load.")

    except Exception as e:
        logger.error(f"Error in example usage: {e}")
