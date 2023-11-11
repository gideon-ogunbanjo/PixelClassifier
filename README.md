# PixelClassifier
PixelClassifier - Image Classification Model using CNN
## Overview
PixelClassifier is an image classification project that leverages Convolutional Neural Networks (CNNs) to accurately classify images from the CIFAR-10 dataset. The model is implemented using TensorFlow and Keras, providing a robust solution for image classification tasks.

### Features:
- Dataset: Utilizes the CIFAR-10 dataset, consisting of 60,000 32x32 color images in 10 different classes.
- Neural Network Architecture: PixelClassifier employs a CNN architecture with convolutional and pooling layers, followed by dense layers for effective image classification.
- Normalization: The training and testing data are normalized to ensure optimal model performance.
- Class Labels: The model is trained to recognize and classify images into the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- Evaluation Metrics: Utilizes accuracy metrics and a detailed classification report for model evaluation.

> I just used this to master CNN with Tensorflow. Here are my callouts:
### CNN Architecture:
- Sequential Model: The Sequential model in Keras represents a linear stack of layers. In this case, layers are added sequentially to the model.
- Convolutional Neural Network (CNN) Layers:
- Conv2D Layer (with ReLU activation): This layer performs convolutional operations on the input data. It uses 32 filters of size (3, 3) with a Rectified Linear Unit (ReLU) activation function. ReLU is chosen for its ability to introduce non-linearity, allowing the network to learn complex patterns.
- MaxPooling2D Layer: This layer performs max pooling operation to down-sample the spatial dimensions. It helps in reducing the computational complexity and the number of parameters in the network.
- Conv2D Layer (with ReLU activation): Another convolutional layer with 64 filters and ReLU activation.
- MaxPooling2D Layer: Another max pooling layer.
- Dense Layers:
- Flatten Layer: Flattens the input into a one-dimensional array. This is required before passing the data to the dense layers.
- Dense Layer (with ReLU activation): A fully connected layer with 64 neurons and ReLU activation. ReLU is commonly used in hidden layers for its computational efficiency and ability to mitigate the vanishing gradient problem.
- Dense Layer (with Softmax activation): The final layer with 10 neurons (matching the number of classes in CIFAR-10) and softmax activation. Softmax is used for multi-class classification as it converts the raw output scores into probability distributions.

### Creator: 
Gideon Ogubanjo