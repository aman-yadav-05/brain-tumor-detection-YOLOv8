from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
# === CONFIGURATION ===
MODEL_PATH = 'brain_tumor_results_corrected/yolov8_tumor_detection/weights/best.pt'
IMAGE_PATH = 'test_images/pred32.jpg'  
CONFIDENCE_THRESHOLD = 0.5

# === LOAD MODEL ===
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

# === RUN INFERENCE ===
print("[INFO] Running inference...")
results = model(source=IMAGE_PATH, conf=CONFIDENCE_THRESHOLD, save=True)

# === GET RESULT PATH ===
output_dir = results[0].save_dir 
output_path = os.path.join(output_dir, os.path.basename(IMAGE_PATH))

# Read the image using OpenCV
img = cv2.imread(output_path)

# Convert BGR (OpenCV default) to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.title("Predicted Tumor Detection")
plt.axis('off')  # Hides axes
plt.show()

print(f"[✅] Result saved to: {output_path}")
