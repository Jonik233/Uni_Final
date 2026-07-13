# Object Detection and Classification

## 1. Project Overview

The system performs two main tasks:

1. **Frame classification** based on the presence or absence of airplanes.
2. **Aircraft localization** and its classification into civilian or military categories.

For the first step, the **ResNet34** CNN was used; for the second, the **YOLOv8s** detector from Ultralytics. Both models were fine-tuned using **PyTorch**.

---

## 2. Data Description

A dataset consisting of images of various airfield types (civilian and military) with annotated labels was used for training and testing the models:

* **Class**: Presence / absence of an aircraft in the frame.
* **Objects**: Localization of aircraft and their classification into military or civilian types.

![Data Samples](metrics/data_samples.png)

---

## 3. Image Classification: ResNet34

### Architecture

ResNet34 is a deep convolutional neural network with 34 layers that utilizes residual connections to prevent the vanishing gradient problem. For our specific task, it features:

* An initial image processing layer with a resolution of 224x224.
* Four blocks of residual modules.
* A final fully-connected layer with 2 neurons (classes:
  * "Aircraft present"
  * "Aircraft absent")

The network was fine-tuned on a custom dataset starting with pre-trained ImageNet weights.

### Performance Metrics

![ResNet34 Metrics](metrics/resnet_metrics.png)

---

## 4. Localization and Classification: YOLOv8s

### Architecture

YOLOv8s is a modern variant of the "You Only Look Once" detector family, offering high speed and accuracy:

* **Backbone**: CSPDarknet
* **Neck**: PANet for multi-scale feature aggregation
* **Head**: Predicts bounding box coordinates and object classes (military/civilian)

The network was fine-tuned using pre-trained Ultralytics weights on a custom dataset of annotated airfield images.

### Performance Metrics

![YOLOv8 Metrics](metrics/yolo_metrics.png)

---

## 5. Test Results

This section presents visual examples of aircraft detection and classification results obtained on the test image set.

![Test Results](metrics/test_results.png)
