# OCR-for-Handwritten-Medical-Prescription-Images
Project Overview:

In this project, I build a handwritten character recognition system using Convolutional Neural Networks (CNNs). The goal is to recognize characters from two datasets: MNIST and A-Z Handwritten Data. The datasets are combined, preprocessed, and used to train a CNN model.

Technologies Used:

TensorFlow and Keras:

TensorFlow, a popular machine learning framework, is used for building and training the neural network.
Keras, an open-source deep learning API running on top of TensorFlow, simplifies the model-building process.
Image Data Preprocessing:

Image data augmentation techniques are applied using Keras's ImageDataGenerator. This includes rescaling, rotation, width/height shifting, shear, zoom, and horizontal flipping.
Neural Network Architecture:

The model architecture involves convolutional layers (Conv2D), max-pooling layers, and densely connected layers (Dense). The final layer uses softmax activation for multi-class classification.

Data Handling:

MNIST and A-Z Handwritten Data are loaded and preprocessed. The datasets are combined, labels are adjusted, and the data is split into training and testing sets.
MNIST DATASET Link : http://yann.lecun.com/exdb/mnist/index.html

A-Z Handwritten DATASET Link : https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

Training and Evaluation:

The model is compiled using the RMSprop optimizer and sparse categorical crossentropy loss function. It is then trained using the prepared datasets, and the training progress is evaluated using accuracy metrics.
Model Persistence:

Once trained, the model is saved for future use, likely for making predictions on new handwritten characters.
