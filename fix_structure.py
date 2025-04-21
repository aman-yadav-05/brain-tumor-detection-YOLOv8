import os
import shutil
from glob import glob

def move_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    for txt_file in glob(os.path.join(image_dir, "*.txt")):
        shutil.move(txt_file, os.path.join(label_dir, os.path.basename(txt_file)))

# Paths (adjust if needed)
train_images = "brain_tumor_yolo/images/train"
val_images = "brain_tumor_yolo/images/val"
train_labels = "brain_tumor_yolo/labels/train"
val_labels = "brain_tumor_yolo/labels/val"

move_labels(train_images, train_labels)
move_labels(val_images, val_labels)

print("✅ Labels moved to correct directories!")
