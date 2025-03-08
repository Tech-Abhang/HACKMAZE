import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

# Function to extract MFCC features
def extract_mfcc(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Function to load the dataset
def load_dataset(dataset_path):
    features = []
    labels = []
    for word in os.listdir(dataset_path):
        word_path = os.path.join(dataset_path, word)
        if not os.path.isdir(word_path):  # Skip if it's not a directory
            continue
        for file_name in os.listdir(word_path):
            file_path = os.path.join(word_path, file_name)
            if file_name.startswith('.'):  # Skip hidden files/folders
                continue
            try:
                mfccs = extract_mfcc(file_path)
                features.append(mfccs)
                labels.append(word)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return np.array(features), np.array(labels)

# Function to create a 1D CNN model
def create_kws_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(8, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Main script
if __name__ == "__main__":
    # Load dataset
    dataset_path = "data/train_dataset"  # Path to training dataset
    features, labels = load_dataset(dataset_path)
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save the label encoder classes
    np.save("models/label_encoder_classes.npy", label_encoder.classes_)
    
    # Reshape features for CNN input (no need to add extra dimension)
    # features shape: (num_samples, num_frames, num_mfcc_coefficients)
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    # Build the model
    input_shape = X_train.shape[1:]  # Shape: (num_frames, num_mfcc_coefficients)
    num_classes = len(label_encoder.classes_)
    model = create_kws_model(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save("models/kws_model.h5")
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open("models/kws_model.tflite", "wb") as f:
        f.write(tflite_model)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")