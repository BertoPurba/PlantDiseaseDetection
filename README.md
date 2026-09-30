# Plant Disease Detection

A deep learning project for classifying plant diseases from leaf images using MobileNetV2.

## Overview

This project uses transfer learning with **MobileNetV2** to classify plant leaf images into **38 different classes**.

The model is trained using the **PlantVillage dataset** and implemented with TensorFlow and Keras.

A Streamlit application is also included to demonstrate the model through an image upload interface.

## Dataset

The project uses the PlantVillage dataset containing images of healthy and diseased plant leaves.

* Training images: 43,444
* Validation images: 10,861
* Number of classes: 38
* Image size: 160 × 160 pixels

## Classes

The model recognizes 38 plant conditions, including:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

Both healthy and diseased conditions are included.

## Model

The project uses **MobileNetV2** with ImageNet pre-trained weights.

The base model is frozen and a classification layer is added on top:

```text
Input Image
     ↓
Data Augmentation
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Layer (38 classes)
     ↓
Softmax Prediction
```

## Training

Training configuration:

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Model             | MobileNetV2                     |
| Image Size        | 160 × 160                       |
| Batch Size        | 32                              |
| Number of Classes | 38                              |
| Optimizer         | Adam                            |
| Learning Rate     | 0.001                           |
| Epochs            | 3                               |
| Loss Function     | Sparse Categorical Crossentropy |

Class weights were also used to help handle class imbalance in the training dataset.

## Results

The model achieved:

| Metric                |     Result |
| --------------------- | ---------: |
| Validation Accuracy   | **91.74%** |
| Validation Loss       | **0.2501** |
| Correct Predictions   |  **9,964** |
| Incorrect Predictions |    **897** |
| Validation Images     | **10,861** |

The confusion matrix and F1-score analysis were used to evaluate performance across the 38 classes.

Some of the more difficult classes were:

* Tomato Early Blight
* Tomato Target Spot
* Tomato Spider Mites
* Tomato Septoria Leaf Spot
* Tomato Mosaic Virus

## Project Structure

```text
PlantDiseaseDetection/
│
├── dataset/
│   └── PlantVillage/
│       ├── train/
│       └── val/
│
├── models/
│   └── plant_disease_mobilenetv2.keras
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

## Streamlit Demo

The project includes a Streamlit application where users can upload a plant leaf image and receive a predicted disease class.

The application displays:

* Uploaded image
* Predicted class
* Prediction confidence

## How to Run

Clone the repository:

```bash
git clone https://github.com/BertoPurba/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Technologies

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

## Author

**Berto Jdoyvan Purba**

Data Science Student — Telkom University

GitHub: https://github.com/BertoPurba
