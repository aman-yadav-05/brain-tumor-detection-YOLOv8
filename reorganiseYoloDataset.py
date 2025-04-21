import os
import shutil

# Base dataset directory
SOURCE_DIR = "brain-tumor-dataset-all"
DEST_DIR = "brain_tumor_yolo"

# Define YOLO structure
IMAGE_DIRS = {
    "train": os.path.join(DEST_DIR, "images", "train"),
    "val": os.path.join(DEST_DIR, "images", "val")
}

# Make sure destination folders exist
for folder in IMAGE_DIRS.values():
    os.makedirs(folder, exist_ok=True)

# Class list (optional, but useful for naming)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Function to move and rename images
def reorganize_images(split):  # 'train' or 'val'
    for cls_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(SOURCE_DIR, "Training" if split == "train" else "Testing", class_name)
        dest_dir = IMAGE_DIRS[split]

        count = 1
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                new_filename = f"{class_name}_{count:03d}.jpg"
                src_path = os.path.join(class_dir, filename)
                dst_path = os.path.join(dest_dir, new_filename)

                shutil.copy2(src_path, dst_path)
                print(f"[{split.upper()}] Copied {filename} → {new_filename}")
                count += 1

# Run for both train and val sets
reorganize_images("train")
reorganize_images("val")

print("\n✅ Dataset reorganized in YOLO format!")
