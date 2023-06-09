from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained VGG-16 model without the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False)

# Create a new model that outputs the last convolutional layer's output
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


def img_from_disk(img_path: str):
    # VGG-16 expects images of size 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    return img


def extract_features(img_path, get_img=img_from_disk):
    img = get_img(img_path)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    features = model.predict(x)

    # Apply global average pooling
    pooled_features = np.mean(features, axis=(1, 2))
    return pooled_features


# Example usage

