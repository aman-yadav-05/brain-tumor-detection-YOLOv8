import os
import shutil
from glob import glob

# 🛠️ Customize this
BASE_DIR = "brain_tumor_dataset"
RAW_IMAGE_DIR = "Training"  # or whatever root folder contains 'glioma/', 'no_tumor/', etc.
VAL_IMAGE_DIR = "Validation"  # your validation set

# ✅ YOLO structure
paths = {
    "images/train": os.path.join(BASE_DIR, "images/train"),
    "images/val": os.path.join(BASE_DIR, "images/val"),
    "labels/train": os.path.join(BASE_DIR, "labels/train"),
    "labels/val": os.path.join(BASE_DIR, "labels/val"),
}

# 📁 Create directories
for path in paths.values():
    os.makedirs(path, exist_ok=True)

# 🔄 Move images and labels
def move_images_and_labels(source_img_dir, dest_img_dir, dest_lbl_dir):
    image_exts = ['*.png', '*.jpg', '*.jpeg']
    for ext in image_exts:
        for img_path in glob(os.path.join(source_img_dir, "**", ext), recursive=True):
            filename = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(filename)[0]
            label_path = os.path.join(os.path.dirname(img_path), f"{name_wo_ext}.txt")

            # Move image
            shutil.copy2(img_path, os.path.join(dest_img_dir, filename))

            # Move label if exists (no-tumor may not have one)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(dest_lbl_dir, f"{name_wo_ext}.txt"))
            else:
                # Empty file for no-tumor
                open(os.path.join(dest_lbl_dir, f"{name_wo_ext}.txt"), 'w').close()

# 🧠 Organize train and val
move_images_and_labels(RAW_IMAGE_DIR, paths["images/train"], paths["labels/train"])
move_images_and_labels(VAL_IMAGE_DIR, paths["images/val"], paths["labels/val"])

print("✅ Dataset prepared in YOLO format!")
