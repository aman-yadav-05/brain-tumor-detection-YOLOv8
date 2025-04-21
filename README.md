# 🧠 Brain Tumor Detection using YOLOv8 + Flask

A web-based application that allows users to upload brain MRI images and receive instant predictions on whether a **tumor is present or not** using a pre-trained **YOLOv8 model**.

> Built with ❤️ using Flask, YOLOv8, OpenCV, and Tailwind CSS for a smooth and user-friendly experience.

---

## 🚀 Features

- 🖼 Upload MRI images in JPG/PNG format
- ⚙️ Uses YOLOv8 model for real-time detection
- 📦 Logs predictions to a local SQLite database
- 📈 View detection history in a `logs.html` dashboard
- 📋 Displays possible symptoms based on prediction
- 💡 Friendly reminders and error handling built-in

---

## 📸 Sample Output

![Prediction Screenshot](static/demo/sample_prediction.png)

---

## 🛠 Technologies Used

| Tech           | Purpose                            |
|----------------|------------------------------------|
| Flask          | Web framework                      |
| YOLOv8 (Ultralytics) | Object detection model        |
| OpenCV         | Image processing                   |
| SQLite         | Local database logging             |
| HTML/CSS       | Frontend                           |
| Tailwind CSS   | Styling & responsive UI            |
| Jinja2         | HTML templating                    |

---

## 🧰 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/brain-tumor-detector-flask.git
cd brain-tumor-detector-flask
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Add Your YOLOv8 Model
Download or train a YOLOv8 model and place the weights at:
```bash
brain_tumor_results_corrected/yolov8_tumor_detection/weights/best.pt
```
### 4. Run the App
```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser.

### 🗃 Database Logging (SQLite)
Each prediction is saved with:

Filename

Label (tumor/no_tumor)

Confidence score

Timestamp


### 📁 Project Structure
```pgsql
├── app.py
├── templates/
│   ├── index.html
│   ├── result.html
│   └── logs.html
├── static/
│   ├── uploads/
│   ├── results/
│   └── demo/
├── database.db
├── requirements.txt
└── README.md
```
