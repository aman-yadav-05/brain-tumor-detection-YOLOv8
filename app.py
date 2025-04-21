from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os
import cv2
from flask_sqlalchemy import SQLAlchemy
import uuid
import matplotlib.pyplot as plt
import matplotlib
from models import db, PredictionLog



matplotlib.use('Agg')  # Use Agg backend for headless environments

# === CONFIGURATION ===
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'brain_tumor_results_corrected/yolov8_tumor_detection/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.5

# === FLASK SETUP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)


# === LOAD MODEL ===
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

SYMPTOMS = {
    "tumor": [
        "Headaches that worsen in the morning",
        "Seizures or convulsions",
        "Memory loss or confusion",
        "Vision or speech issues"
    ],
    "no_tumor": [
        "No symptoms detected. This image likely doesn't show a tumor."
    ]
}
# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save file
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run inference
    results = model(source=filepath, conf=CONFIDENCE_THRESHOLD, save=True)
    output_dir = results[0].save_dir
    output_img_path = os.path.join(output_dir, filename)

    
# === Save the predicted image to your result folder ===
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, cv2.imread(output_img_path))

    # === Extract prediction label and confidence ===
        # === Check for detections ===
    if results[0].boxes is None or len(results[0].boxes) == 0:
        pred_label = "No tumor detected"
        confidence = 0.0
        # confidence = float(results[0].boxes.conf[0])

    else:
        pred_label = results[0].names[int(results[0].boxes.cls[0])]
        confidence = float(results[0].boxes.conf[0])
    # === Save to database ===
    new_log = PredictionLog(
        filename=filename,
        label=pred_label,
        confidence=round(confidence * 100, 2)
    )
    db.session.add(new_log)
    db.session.commit()

    #getting symptoms if tumor detected
    symptoms = SYMPTOMS.get(pred_label.lower(), ["No noticeable symptoms"])

    # === Render the result page ===
    return render_template("result.html",
                          result_image=os.path.join("static", "results", filename),
                          label=pred_label,
                          confidence=round(confidence * 100, 2),
                          symptoms=symptoms)


@app.route('/logs')
def view_logs():
    logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    return render_template('logs.html', logs=logs)

# === RUN APP ===
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    with app.app_context():
      db.create_all()
    app.run(debug=True)
