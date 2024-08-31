# Facial Emotion Recognition using CNN with Real-Time Prediction

## Project Overview
This project involves developing a Convolutional Neural Network (CNN) for recognizing facial emotions in images. The model is trained on the RAF-DB dataset, which contains a wide range of facial expressions. It classifies images into seven emotion categories: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. The project also integrates real-time emotion detection using a Haar Cascade classifier, allowing the model to predict emotions from live video input.

## Introduction
Facial emotion recognition is a critical component in human-computer interaction, with applications ranging from mental health monitoring to interactive entertainment. This project uses a CNN to classify facial expressions into seven distinct emotions. The model is further integrated with a Haar Cascade classifier for real-time face detection, enabling live emotion recognition via webcam.

## Dataset
The model is trained on the RAF-DB (Real-world Affective Faces Database) dataset, which contains thousands of annotated facial images categorized into seven emotion classes:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

The dataset is preprocessed to normalize the images and prepare them for training.

## Installation
To run this project, ensure you have Python 3.x installed, along with the following libraries:

- TensorFlow or PyTorch
- Keras (if using TensorFlow)
- pandas
- numpy
- matplotlib
- scikit-learn
- OpenCV

Install the required packages using pip:
```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn opencv-python
