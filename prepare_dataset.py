import os
import shutil
import random

print("Script started...")

# --- CHOOSE WHICH DATASET TO SPLIT ---
# Change this to 'yawn' to split the yawn dataset
DATASET_TO_SPLIT = 'yawn'
# -------------------------------------


# --- Configuration ---
BASE_DIR = os.path.join('datasets', f'{DATASET_TO_SPLIT}_data')
SOURCE_DIR = os.path.join(BASE_DIR, 'train')

VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Dynamically get class names from the train folder
try:
    CLASSES = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    if not CLASSES:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"Error: Could not find class folders inside '{SOURCE_DIR}'.")
    print("Please make sure your 'train' folder is set up correctly before running.")
    exit()

print(f"Found classes for '{DATASET_TO_SPLIT}' dataset: {CLASSES}")
# ---------------------


def create_and_split_data():
    """
    Creates validation and test directories and splits the data.
    """
    print(f"\n--- Splitting '{DATASET_TO_SPLIT}' dataset ---")
    validation_dir = os.path.join(BASE_DIR, 'validation')
    test_dir = os.path.join(BASE_DIR, 'test')

    # Start fresh by deleting old directories if they exist
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(validation_dir)
    os.makedirs(test_dir)

    for cls in CLASSES:
        os.makedirs(os.path.join(validation_dir, cls))
        os.makedirs(os.path.join(test_dir, cls))

    for cls in CLASSES:
        print(f"\nProcessing class: {cls}")
        source_class_dir = os.path.join(SOURCE_DIR, cls)

        all_files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
        random.shuffle(all_files)

        num_files = len(all_files)
        num_validation = int(num_files * VALIDATION_SPLIT)
        num_test = int(num_files * TEST_SPLIT)

        validation_files = all_files[:num_validation]
        test_files = all_files[num_validation : num_validation + num_test]

        print(f"Total images: {num_files} | Moving {len(validation_files)} to validation, {len(test_files)} to test.")

        # Move validation files
        for file_name in validation_files:
            shutil.move(os.path.join(source_class_dir, file_name), os.path.join(validation_dir, cls, file_name))

        # Move test files
        for file_name in test_files:
            shutil.move(os.path.join(source_class_dir, file_name), os.path.join(test_dir, cls, file_name))

    print(f"\n'{DATASET_TO_SPLIT}' dataset splitting complete!")


if __name__ == "__main__":
    create_and_split_data()