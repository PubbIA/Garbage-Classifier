# Garbage Classification using MobileNetV2

This project demonstrates garbage classification using transfer learning with MobileNetV2. The dataset consists of images categorized into several classes: cardboard, glass, metal, paper, plastic, and trash. The goal is to build a model that can classify these images accurately.

## Data Preprocessing

The dataset is loaded and preprocessed using OpenCV for image handling and TensorFlow/Keras utilities for data augmentation. Images are resized to 224x224 pixels, which is the input size required for MobileNetV2.

## Augmentation

Data augmentation techniques are applied to increase the diversity of the dataset, including rotation, shifting, shearing, zooming, and flipping.

## Model Training

### Transfer Learning with MobileNetV2

- MobileNetV2 is used as the base model for transfer learning.
- The base model is frozen, and a custom classification head is added on top.
- The model is compiled with Adam optimizer, Sparse Categorical Crossentropy loss function, and Sparse Categorical Accuracy metric.

### Training and Evaluation

- The data is split into training (80%) and validation (20%) sets.
- The model is trained with early stopping to prevent overfitting.
- Training progress is monitored with loss and accuracy metrics.

### Results

- After training, the model achieves an accuracy of [insert your accuracy here] on the test set.
- Training and validation loss/accuracy curves are plotted for analysis.

## Prediction

- The trained model can predict the class of a given image from the test set.
- Example predictions are demonstrated with corresponding images.

## Repository Contents

- `garbage_classification.ipynb`: Jupyter notebook containing the entire code and explanation.
- `garbage_classifier.h5`: Trained model file.

## Usage

To run the notebook:

1. Install required dependencies (`opencv-python`, `numpy`, `matplotlib`, `tensorflow`, `scikit-learn`).
2. Open and execute the notebook step-by-step in a Jupyter environment.

## License

This project is licensed under the [GPL License](LICENSE).
