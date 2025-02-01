# Custom YOLO Training and Deployment Guide

## 1️⃣ Introduction
This repository provides a complete setup for training, validating, and testing custom YOLO models using a well-structured dataset. It supports multiple YOLO model types, including detection, pose estimation, and classification.

Additionally, this repository allows you to create labeled datasets using **LabelImg**, making it easy to generate bounding box annotations for training custom models.

---

## 2️⃣ Repository Structure
```
├── data/
│   ├── train/   # Training images and labels
│   ├── valid/   # Validation images and labels
│   ├── test/    # Test images and labels (Optional)
│   ├── data.yaml  # Configuration file specifying paths and class names
│
├── notebooks/
│   ├── custom_yolo_training.ipynb  # Notebook for training YOLO models
│   ├── yolo_detection.ipynb         # Notebook for object detection using trained models
│
├── models/
│   ├── yolov8n.pt  # YOLOv8 nano model (smallest & fastest)
│   ├── yolov8m.pt  # YOLOv8 medium model (balanced)
│   ├── yolov8l.pt  # YOLOv8 large model (most accurate)
│   ├── yolov8x.pt  # YOLOv8 extra-large model (best accuracy)
│
└── README.md  # This guide
```

---

## 3️⃣ Setting Up the Environment

### **Install Required Libraries**
Ensure you have Python **3.8 or above** installed, then run:
```bash
pip install ultralytics opencv-python numpy pyyaml matplotlib
```

For Jupyter Notebook:
```bash
pip install notebook
```

---

## 4️⃣ Preparing the Dataset

### **Organizing Dataset**
Your dataset should be structured as follows:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│
├── valid/
│   ├── images/
│   ├── labels/
│
├── test/   # (Optional)
│   ├── images/
│   ├── labels/
│
└── data.yaml  # Configuration file
```

### **Creating `data.yaml` File**
Create a `data.yaml` file inside the `data/` folder with the following content:
```yaml
train: ./data/train/images  # Path to training images
val: ./data/valid/images    # Path to validation images
test: ./data/test/images    # Path to test images (optional)

nc: 3  # Number of classes
names: ['car', 'truck', 'bus']  # Class names
```

---

## 5️⃣ Labeling Images using LabelImg

### **Installing LabelImg**

#### **Method 1: Install via pip**
```bash
pip3 install labelImg
```
Run LabelImg:
```bash
labelImg
```

#### **Method 2: Build from Source (Linux/Ubuntu/Mac)**
```bash
sudo apt-get install pyqt5-dev-tools
pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
```

### **How to Label Images**
1. Open **LabelImg**
2. Click **Open Dir** → Select the `train/images` folder
3. Click **Change Save Dir** → Set it to `train/labels`
4. Draw bounding boxes around objects
5. Click **Save** (labels will be saved as `.txt` files in `train/labels/`)

More details: [LabelImg GitHub](https://github.com/HumanSignal/labelImg)

---

## 6️⃣ Training the Custom YOLO Model

### **Step 1: Train YOLOv8 Model**
Run the following command inside your Jupyter Notebook:
```python
from ultralytics import YOLO

# Load YOLOv8 model (Choose model: yolov8n, yolov8m, yolov8l, yolov8x)
model = YOLO("yolov8m.pt")  # Using medium model

# Train the model
model.train(data="data/data.yaml", epochs=50, imgsz=640, batch=16, workers=4)
```

### **Step 2: Validate Model**
```python
results = model.val()
```

### **Step 3: Run Detection on Test Images**
```python
model.predict("data/test/images", save=True, conf=0.5)
```

---

## 7️⃣ Running YOLO for Object Detection, Pose, or Classification

### **For Object Detection**
```python
model = YOLO("best.pt")
model.predict("sample_image.jpg", save=True, conf=0.5)
```

### **For Pose Estimation**
```python
model = YOLO("yolov8m-pose.pt")
model.predict("sample_image.jpg", save=True)
```

### **For Image Classification**
```python
model = YOLO("yolov8m-cls.pt")
model.predict("sample_image.jpg", save=True)
```

---

## 8️⃣ Exporting Trained Model

### **Convert to ONNX for Deployment**
```python
model.export(format="onnx")
```

### **Convert to TensorRT for Faster Inference**
```python
model.export(format="engine")
```

---

## 9️⃣ Summary
- **Label images** using LabelImg
- **Train custom YOLO models** with `data.yaml`
- **Validate and test models** using `model.val()`
- **Use YOLO for detection, pose estimation, and classification**
- **Deploy model using ONNX or TensorRT**



