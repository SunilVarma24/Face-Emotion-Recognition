# main.py
from src.data import load_images_labels, shuffle_and_encode
from src.visualization import visualize_images, plot_training_history, plot_confusion
from src.augmentation import create_datagen, visualize_augmentation
from src.model import build_cnn_model, compile_and_train, evaluate_model
from src.realtime import run_realtime_emotion_recognition
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # 1. Read the image dataset
    train_path = '/kaggle/input/faceer2/train'
    test_path = '/kaggle/input/faceer2/test'
    train_images, train_labels = load_images_labels(train_path)
    test_images, test_labels = load_images_labels(test_path)
    
    # Shuffle and encode labels
    train_images, train_onehot_encoded, test_images, test_onehot_encoded, train_labels, test_labels = \
        shuffle_and_encode(train_images, train_labels, test_images, test_labels)
    
    # 2. Visualize train and test images
    print("Visualizing Train Images:")
    visualize_images(train_images, train_labels, num_samples=5)
    print("Visualizing Test Images:")
    visualize_images(test_images, test_labels, num_samples=5)
    
    # 3. Data Augmentation
    # Ensure that training images have channel dimension (for grayscale images)
    x_train_aug = train_images
    if x_train_aug.ndim == 3:
        x_train_aug = np.expand_dims(x_train_aug, axis=-1)
    train_datagen, x_train_aug = create_datagen(x_train_aug, train_onehot_encoded, batch_size=32)
    visualize_augmentation(train_datagen, img_shape=(100,100), num_samples=8)
    
    # 4. Train-Val Split on original training data
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_onehot_encoded, test_size=0.08, random_state=0)
    
    print("Train data shape:", x_train.shape)
    print("Validation data shape:", x_val.shape)
    
    # 5. Define and build the CNN Model
    # Expand dims for grayscale channel if necessary
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)
        test_images_exp = np.expand_dims(test_images, axis=-1)
    else:
        test_images_exp = test_images
    model = build_cnn_model(input_shape=(100,100,1), num_classes=7)
    
    # Calculate class weights
    class_labels = np.argmax(y_train, axis=1)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    # 6. Model Training
    history, model = compile_and_train(model, train_datagen, x_val, y_val, class_weight_dict, epochs=100, batch_size=128)
    plot_training_history(history)
    
    # 7. Model Evaluation
    test_true, test_pred, test_loss, test_accuracy = evaluate_model(model, test_images_exp, test_onehot_encoded)
    print("Classification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(test_true, test_pred))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_true, test_pred)
    plot_confusion(cm)
    
    # Save the model
    model.save('FER.h5')
    print("Model saved as FER.h5")
    
    # 8. Real-Time Prediction (Uncomment to run real-time prediction)
    # Uncomment the following lines to run real-time prediction. Ensure that a webcam is available.
    # emotion_model_path = 'FER.h5'
    # EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    # run_realtime_emotion_recognition(emotion_model_path, EMOTIONS)

if __name__ == "__main__":
    main()
