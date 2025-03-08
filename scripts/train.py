import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
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
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(word)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return np.array(features), np.array(labels)

# Main script
if __name__ == "__main__":
    # Load training dataset
    train_dataset_path = "data/train_dataset"  # Path to training dataset
    X_train, y_train = load_dataset(train_dataset_path)
    
    # Load testing dataset
    test_dataset_path = "data/test_dataset"  # Path to testing dataset
    X_test, y_test = load_dataset(test_dataset_path)
    
    # Normalize features
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)  # Use the same encoder for testing data
    
    # Save the label encoder classes
    np.save("models/label_encoder_classes.npy", label_encoder.classes_)
    
    # Reshape features for CNN input
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Build the model
    input_shape = X_train.shape[1:]
    num_classes = len(label_encoder.classes_)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save("models/voice_recognition_model.h5")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")