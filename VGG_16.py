from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


# Load the pre-trained VGG-16 model without the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False)

# Create a new model that outputs the last convolutional layer's output
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def img_from_disk(img_path:str):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG-16 expects images of size 224x224
    return img


def extract_features(get_img,img_path):
    img = get_img(img_path)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return  model.predict(x)