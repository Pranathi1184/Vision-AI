# Vision-AI
# CIFAR-10 Image Recognition Using Convolutional Neural Networks
# Overview
  This project focuses on building and evaluating a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The CIFAR-10   dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to develop a model    that can accurately classify these images into their respective categories.

# Table of Contents
  - Project Objective
  - Dataset
  - Approach
  - Model Architecture
  - Results

# Project Objective
The primary objective of this project is to create a robust image classification model that can accurately identify objects in images from the CIFAR-10 dataset. The model aims to achieve high accuracy and generalization through the use of data augmentation and a well-structured CNN architecture.

# Dataset
CIFAR-10: The dataset consists of 60,000 images divided into 10 classes:
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
Each class contains 6,000 images, and the images are 32x32 pixels in size.

# Approach
- Data Loading: The CIFAR-10 dataset is loaded using TensorFlow's built-in dataset loader.
- Data Preprocessing: Images are normalized to a range of [0, 1] to improve model training.
- Data Augmentation: Techniques such as rotation, width/height shifts, and horizontal flips are applied to enhance the training dataset and improve model generalization.
- Model Development: A CNN is built using TensorFlow and Keras, consisting of:
    - Convolutional layers with ReLU activation
    - Batch Normalization
    - MaxPooling layers
    - Dropout layers to prevent overfitting
    - A final softmax layer for multi-class classification
- Model Training: The model is trained using the augmented dataset for 20 epochs, with validation on a separate test set.
- Model Evaluation: The model's performance is evaluated using accuracy, classification reports, and confusion matrices.

# Model Architecture
The CNN architecture consists of the following layers:
  - Input Layer: Accepts 32x32x3 images.
  - Convolutional Layers: Three convolutional layers with increasing filter sizes (32, 64, 128).
  - Batch Normalization: Applied after each convolutional layer.
  - MaxPooling Layers: Reduces spatial dimensions.
  - Flatten Layer: Converts 2D matrices to 1D vectors.
  - Dense Layers: Fully connected layers with dropout for regularization.
  - Output Layer: Softmax activation for multi-class classification.
  - 
# Results
Training Accuracy: Achieved a maximum training accuracy of approximately 90%.
Validation Accuracy: The model reached a validation accuracy of around 85%.
Test Accuracy: The final test accuracy was approximately 84%.
Classification Report: Detailed precision, recall, and F1-score for each class were generated.
Confusion Matrix: Visual representation of model predictions vs. actual labels.
