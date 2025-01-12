import os
from app.model_training_loading import super_resolve_with_multiplier, load_or_train_model

if __name__ == "__main__":
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stored_models",
                              "trained_super_resolution_model.keras")
    INPUT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_assets",
                                    "test_img.png")
    OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_assets",
                                     "output_super_resolved_image.jpg")

    # Load or train the model
    model = load_or_train_model(MODEL_PATH, DATASET_PATH)

    # Multiplier for scaling resolution
    MULTIPLIER = int(input("Enter a multiplier (ex 2, 3, 4): "))

    # Super-resolve an image while preserving aspect ratio
    super_resolve_with_multiplier(
        model, INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, multiplier=MULTIPLIER, target_size=(64, 64)
    )
