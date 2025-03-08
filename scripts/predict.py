import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Function to preprocess new audio
def preprocess_new_audio(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    mfccs = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)  # Normalize
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    return mfccs

# Function to predict a word
def predict_word(model, label_encoder, file_path):
    processed_audio = preprocess_new_audio(file_path)
    predictions = model.predict(processed_audio)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_word = label_encoder.inverse_transform(predicted_label)
    return predicted_word[0]

# Main script
if __name__ == "__main__":
    # Load the model
    model = tf.keras.models.load_model("models/voice_recognition_model.h5")
    
    # Load the label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("models/label_encoder_classes.npy", allow_pickle=True)
    
    # Path to the new audio files
    test_audio_folder = "data/test_audio"
    
    # Get list of audio files in the test folder
    test_files = [os.path.join(test_audio_folder, f) for f in os.listdir(test_audio_folder) if f.endswith('.wav')]
    
    # Predict for each audio file
    print("Testing new audio files:")
    for file_path in test_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")