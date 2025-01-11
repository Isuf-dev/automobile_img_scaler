import os

import cv2
import numpy as np
from keras import Input, Model
from keras.src.layers import Conv2D, UpSampling2D
from keras.src.saving.saving_lib import load_model


def load_and_preprocess_data(dataset_path, image_size=(128, 128)):
    print("Loading and preprocessing dataset...")
    images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                images.append(img)
    images = np.array(images, dtype=np.float32) / 255.0
    return images


# Define Super-Resolution Model
def build_super_resolution_model(input_shape=(64, 64, 3)):
    print("Building super-resolution model...")
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_model(model, low_res_images, high_res_images, batch_size=32, epochs=100):
    print("Training the model...")
    model.fit(low_res_images, high_res_images, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model


def save_trained_model(model, model_path):
    print(f"Saving trained model to {model_path}...")
    model.save(model_path)


def load_or_train_model(model_path, dataset_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = load_model(model_path)
    else:
        print("No existing model found. Training a new model...")
        images = load_and_preprocess_data(dataset_path)
        low_res_images = np.array([cv2.resize(img, (64, 64)) for img in images])
        high_res_images = images

        model = build_super_resolution_model()
        model = train_model(model, low_res_images, high_res_images, batch_size=16, epochs=200)
        save_trained_model(model, model_path)
    return model


def super_resolve_with_multiplier(model, image_path, output_path, multiplier=2, target_size=(64, 64)):
    print(f"Super-resolving the image: {image_path}")
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Maintain the aspect ratio during downscaling
    h, w, _ = input_image.shape
    new_h, new_w = target_size
    low_res_image = cv2.resize(input_image, (new_w, new_h))  # Downscale
    low_res_image = np.expand_dims(low_res_image, axis=0).astype(np.float32) / 255.0

    # Perform super-resolution
    high_res_image = model.predict(low_res_image)[0]

    # Upscale to the desired size using the multiplier while preserving aspect ratio
    output_h = h * multiplier
    output_w = w * multiplier
    high_res_image = cv2.resize((high_res_image * 255.0).astype(np.uint8), (output_w, output_h))

    # Convert back to BGR and save
    high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, high_res_image)
    print(f"Super-resolved image saved to {output_path}")
