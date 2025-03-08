import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report  # Add this import
from collections import Counter
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
EarlyStopping = tf.keras.callbacks.EarlyStopping

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
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Main script
if __name__ == "__main__":
    # Ensure the models folder exists
    os.makedirs("models", exist_ok=True)
    
    # Load dataset
    dataset_path = "data/train_dataset"  # Path to training dataset
    features, labels = load_dataset(dataset_path)
    
    # Print dataset statistics
    print(f"Number of samples: {len(features)}")
    print(f"Feature shape: {features[0].shape}")
    print(f"Classes: {np.unique(labels)}")
    print(f"Samples per class: {Counter(labels)}")
    
    # Normalize features
    std_dev = np.std(features, axis=0)
    std_dev[std_dev == 0] = 1.0  # Avoid division by zero
    features = (features - np.mean(features, axis=0)) / std_dev
    
    # Check for NaN or Inf values
    if np.isnan(features).any() or np.isinf(features).any():
        print("Warning: Dataset contains NaN or Inf values. Replacing with zeros.")
        features = np.nan_to_num(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save the label encoder classes
    np.save("models/label_encoder_classes.npy", label_encoder.classes_)
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    # Build the model
    input_shape = X_train.shape[1:]  # Shape: (num_frames, num_mfcc_coefficients)
    print(f"Model input shape: {input_shape}")
    num_classes = len(label_encoder.classes_)
    model = create_kws_model(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Save the model
    model.save("models/kws_model.keras")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")
    
    # Print classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))