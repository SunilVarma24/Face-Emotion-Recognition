# src/augmentation.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_datagen(x_train, y_train, batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Add channel dimension for grayscale images if not present
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
    train_datagen = datagen.flow(x_train, y_train, batch_size=batch_size)
    return train_datagen, x_train

def visualize_augmentation(train_datagen, img_shape=(100,100), num_samples=8):
    batch_x, batch_y = train_datagen.next()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        # Remove channel dimension for display if needed
        img = batch_x[i].reshape(img_shape)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class: {batch_y[i]}')
        ax.axis('off')
    plt.show()
