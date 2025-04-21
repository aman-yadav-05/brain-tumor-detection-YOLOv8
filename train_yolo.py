import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === Configurations ===
data_yaml = "data.yaml"  # Path to your dataset config
model_arch = "yolov8n.pt"                # Model architecture (change as needed)
epochs = 20
imgsz = 320
project = "brain_tumor_results_corrected"
name = "yolov8_tumor_detection"
log_dir = f"runs/train/{name}"

# === Start Training ===
if __name__ == "__main__":
    print("🚀 Starting YOLOv8 Training...")

    # Train the model
    model = YOLO(model_arch)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True
    )

    print("✅ Training Complete!")

    # === Post-Training: Extract and Plot Metrics ===

    # Path to results (YOLO saves the logs in a CSV file)
    results_file = os.path.join(log_dir, "results.csv")
    
    # Check if the results.csv file exists
    if os.path.exists(results_file):
        # Load the metrics from the CSV file
        df = pd.read_csv(results_file)

        # Display metrics in table format
        print("📊 Training and Validation Metrics:")
        print(df)

        # Plotting metrics like loss, accuracy, precision, recall, F1 score
        plt.figure(figsize=(10, 6))

        # Plot Training Loss vs. Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linestyle='--')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Precision, Recall, and F1 Score
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['precision'], label='Precision')
        plt.plot(df['epoch'], df['recall'], label='Recall')
        plt.plot(df['epoch'], df['f1'], label='F1 Score')
        plt.title('Precision, Recall & F1 Score vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()

        # Plot Training Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy')
        plt.title('Training Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Validation Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
        plt.title('Validation Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Optional: Save the plot as an image
        plt.savefig('training_metrics.png')
    
    else:
        print("⚠️ No results.csv file found. Ensure that training is complete and the log directory exists.")

    print("✅ Visualization complete!")
