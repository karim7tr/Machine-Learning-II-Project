# Machine-Learning-II-Project
# Object detection and Classification using Convolutional Neural Networks on CIFAR-10

## Overview

This project explores the impact of different hyperparameters on Convolutional Neural Networks (CNNs) using the CIFAR-10 dataset. Our goal is to optimize the model architecture and preprocessing techniques to enhance the accuracy and efficiency of CNNs in image classification tasks.

## Project Objectives

1. **Hyperparameter Optimization:** Explore various hyperparameter configurations to find the best settings for improving model accuracy and efficiency.
2. **Data Preprocessing and Enhancement:** Use advanced preprocessing and data augmentation techniques to prepare and enhance the dataset for building a robust CNN model.

## Dataset

### CIFAR-10

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 testing images.

## Directory Structure

- `data/`
  - `train/`
  - `validation/`
  - `test/`
- `models/`
  - `model_1.h5`
  - `model_2.h5`
  - `model_3.h5`
  - `model_4.h5`
  - `model_5.h5`
  - `model_6.h5`
  - `model_7.h5`
  - `model_8.h5`
- `notebooks/`
  - `model_1.ipynb`
  - `model_2.ipynb`
  - `model_3.ipynb`
  - `model_4.ipynb`
  - `model_5.ipynb`
  - `model_6.ipynb`
  - `model_7.ipynb`
  - `model_8.ipynb`
- `README.md`

## Data Preprocessing

1. **Normalization:** Pixel values are scaled to the range [0, 1].
2. **One-Hot Encoding:** Class labels are converted into binary vectors using `tf.keras.utils.to_categorical()`.
3. **Data Augmentation:** Techniques like rotation, zooming, and shifting are applied to increase training data variety and model robustness.

## Model Architecture

### General CNN Structure

1. **Input Layer:** 32x32 RGB images.
2. **Convolutional Layers:** Apply filters to extract features.
3. **Activation Functions:** Introduce non-linearity (ReLU).
4. **Pooling Layers:** Reduce spatial dimensions (Max Pooling).
5. **Flatten Layer:** Converts 2D feature maps into a 1D vector.
6. **Dense (Fully Connected) Layers:** Aggregate features to make predictions.
7. **Output Layer:** Softmax activation for classification.

### Detailed Architecture for Each Model

The project includes eight different models, each with unique configurations of convolutional layers, pooling layers, dense layers, activation functions, and regularization techniques.

### Model 1

- 2 Convolutional Layers
- Max Pooling
- Flatten
- 1 Dense Layer
- Softmax Output

### Model 2

- 3 Convolutional Layers
- Max Pooling
- Flatten
- 2 Dense Layers
- Softmax Output

### Model 3

- 4 Convolutional Layers
- Max Pooling
- Flatten
- 2 Dense Layers
- Softmax Output

### Model 4

- 5 Convolutional Layers
- Max Pooling
- Flatten
- 3 Dense Layers
- Softmax Output

### Model 5

- Batch Normalization after each Convolutional Layer
- 3 Convolutional Layers
- Max Pooling
- Flatten
- 2 Dense Layers
- Softmax Output

### Model 6

- Dropout after each Dense Layer
- 4 Convolutional Layers
- Max Pooling
- Flatten
- 2 Dense Layers
- Softmax Output

### Model 7

- Increased Filter Size in Convolutional Layers
- 3 Convolutional Layers
- Max Pooling
- Flatten
- 2 Dense Layers
- Softmax Output

### Model 8

- Combination of Batch Normalization and Dropout
- 4 Convolutional Layers
- Max Pooling
- Flatten
- 3 Dense Layers
- Softmax Output

## Training and Evaluation

### Compilation

- **Optimizer:** Adam with a learning rate of 0.001
- **Loss Function:** Categorical cross-entropy
- **Metrics:** Accuracy

### Training

Models are trained using the `fit()` function with:

- **Batch Size:** Number of samples processed before updating weights.
- **Number of Epochs:** Total iterations over the dataset.
- **EarlyStopping:** To halt training if performance on the validation set doesn’t improve.

### Evaluation

- **Accuracy and Loss:** Measure learning and generalization.
- **Confusion Matrix:** To understand the classification performance.
- **Training and Validation Curves:** Analyze accuracy and loss curves for insights.

## Tools and Libraries

- **Keras:** High-level neural network API for fast prototyping.
- **TensorFlow:** Open-source library for machine learning and deep learning.
- **Matplotlib:** For plotting training and validation metrics.

## Conclusion

This project aims to optimize CNN models for better image classification using the CIFAR-10 dataset. By fine-tuning hyperparameters and applying advanced preprocessing techniques, we developed CNN algorithms that excel in accuracy and generalization.

## How to Use

1. **Clone the repository:**
   ```sh
   git clone https://github.com/karim7tr/Machine-Learning-II-Project.git
   cd Machine-Learning-II-Project
   ```

2. **Install required libraries:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebooks:**
   ```sh
   jupyter notebook notebooks/model_1.ipynb
   ```

4. **Train the models:**
   Follow the instructions in each notebook to train and evaluate the models.

## teamMates.
- Karim Triki
- Ines Haouala
