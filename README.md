# Custom YOLO Training and Deployment Guide

---

## 1️⃣ Introduction
This repository provides a complete setup for training, validating, and testing custom YOLO models using a well-structured dataset. It supports multiple YOLO model types, including detection, pose estimation, and classification.

Additionally, this repository allows you to create labeled datasets using **LabelImg**, making it easy to generate bounding box annotations for training custom models. You can also import datasets directly from **Roboflow**.

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

Additionally, to work with Roboflow, run:
```bash
!pip install roboflow
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

### **Train YOLOv8 Model Using Command Line**
You can also train your YOLO model via the command line. Here's the command for training a model:
```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=224 plots=True
```
This command specifies:
- **task**: Object detection (`detect`)
- **mode**: Training (`train`)
- **model**: The YOLOv8 model to use (you can select `yolov8n.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt`)
- **data**: Path to the `data.yaml` file
- **epochs**: Number of training epochs
- **imgsz**: Image size (224 in this example, can be adjusted based on model requirements)
- **plots**: Set to `True` to generate training plots

---

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

## 9️⃣ Importing Datasets from Roboflow

You can easily import datasets directly from **Roboflow** using the following Python code:

```python
from roboflow import Roboflow

# Initialize the Roboflow API with your API key
rf = Roboflow(api_key="y2BO50RFoRosd6vUrc8r")

# Access the desired project and dataset version
project = rf.workspace("mohamed-el-afia-m2dc3").project("digitraffic")
version = project.version(1)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")
```

This will automatically download the dataset in a format compatible with YOLOv8 for training.

---

## 10️⃣ Summary
- **Label images** using LabelImg
- **Train custom YOLO models** with `data.yaml` (via Python or command line)
- **Validate and test models** using `model.val()`
- **Use YOLO for detection, pose estimation, and classification**
- **Deploy model using ONNX or TensorRT**
- **Import datasets directly from Roboflow** for easier access

