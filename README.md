## 🎯 Objective

To build a **real-time age and gender detector** using OpenCV’s deep learning module (`cv2.dnn`) that can estimate the **approximate age and gender** of a person from an image or webcam stream.

---

## 🧩 About the Project

This project detects **faces**, then predicts **gender (Male/Female)** and **exact age (approximation)** using pre-trained Caffe models.

The models were trained by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html) on the **Adience Dataset**, which contains over **26,000 facial images** across multiple age ranges under real-world conditions.

> ⚙️ Prediction is based on probability-weighted averages, providing a near-exact age instead of a fixed range.

---

## 🧠 Model Files Used

| Type | Description | File |
|------|--------------|------|
| Face Detection | TensorFlow face detection model | `opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt` |
| Age Detection | Caffe model trained on Adience dataset | `age_net.caffemodel`, `age_deploy.prototxt` |
| Gender Detection | Caffe model trained on Adience dataset | `gender_net.caffemodel`, `gender_deploy.prototxt` |

---



### 1 Install dependencies
```bash
python -m pip install opencv-python==4.10.0.84 argparse
```

> Optional: For GPU acceleration, install `opencv-contrib-python` instead.

### 2 Ensure model files exist
Make sure all `.pb`, `.pbtxt`, `.prototxt`, and `.caffemodel` files are in the same directory as `detect.py`.

---

## 🚀 Usage

### 📸 Detect Age & Gender from Image
```bash
python detect.py --image your_image.jpg
```

### 🎥 Real-time Webcam Detection
```bash
python detect.py
```

Press **Q** to quit the webcam window.

---
