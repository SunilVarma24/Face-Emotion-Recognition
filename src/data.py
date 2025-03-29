# src/data.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_images_labels(dataset_path, img_size=100):
    images = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # Load image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Resize image to common size (100x100)
                image = cv2.resize(image, (img_size, img_size))
                # Normalize pixel values
                image = image / 255.0
                images.append(image)
                labels.append(class_name)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def shuffle_and_encode(train_images, train_labels, test_images, test_labels):
    # Shuffle train data
    train_indices = np.random.permutation(len(train_images))
    test_indices = np.random.permutation(len(test_images))
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_label_encoded = label_encoder.fit_transform(train_labels)
    test_label_encoded = label_encoder.transform(test_labels)
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    train_label_encoded = train_label_encoded.reshape(-1, 1)
    test_label_encoded = test_label_encoded.reshape(-1, 1)
    
    train_onehot_encoded = onehot_encoder.fit_transform(train_label_encoded)
    test_onehot_encoded = onehot_encoder.transform(test_label_encoded)
    
    return train_images, train_onehot_encoded, test_images, test_onehot_encoded, train_labels, test_labels
