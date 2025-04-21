import os

# === Folder paths ===
label_base_path = "brain_tumor_yolo/labels"
splits = ["train", "val"]

# === Function to validate YOLO format ===
def is_valid_yolo_label(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return True  # Empty file = non-tumor, still valid
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            if not parts[0].isdigit():
                return False
            try:
                _ = [float(x) for x in parts[1:]]
            except ValueError:
                return False
    return True

# === Loop through both train and val label folders ===
for split in splits:
    folder_path = os.path.join(label_base_path, split)
    print(f"🔍 Checking: {folder_path}")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, filename)

        if not is_valid_yolo_label(file_path):
            print(f"❌ Removing invalid label file: {filename}")
            os.remove(file_path)

print("✅ Clean-up complete! All invalid label files removed.")
