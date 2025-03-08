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
    
    # Test on "up" files
    up_files = [
        "data/voice_dataset/up/up_1.wav",
        "data/voice_dataset/up/up_2.wav",
        "data/voice_dataset/up/up_3.wav",
        "data/voice_dataset/up/up_4.wav",
        "data/voice_dataset/up/up_5.wav"
    ]
    
    print("Testing 'up' files:")
    for file_path in up_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "down" files
    down_files = [
        "data/voice_dataset/down/down_1.wav",
        "data/voice_dataset/down/down_2.wav",
        "data/voice_dataset/down/down_3.wav",
        "data/voice_dataset/down/down_4.wav",
        "data/voice_dataset/down/down_5.wav"
    ]
    
    print("Testing 'down' files:")
    for file_path in down_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Hackmaze" files
    hackmaze_files = [
        "data/voice_dataset/hackmaze/hackmaze_1.wav",
        "data/voice_dataset/hackmaze/hackmaze_2.wav",
        "data/voice_dataset/hackmaze/hackmaze_3.wav",
        "data/voice_dataset/hackmaze/hackmaze_4.wav",
        "data/voice_dataset/hackmaze/hackmaze_5.wav"
    ]
    
    print("Testing 'Hackmaze' files:")
    for file_path in hackmaze_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Triple IT" files
    triple_it_files = [
        "data/voice_dataset/triple_it/triple_it_1.wav",
        "data/voice_dataset/triple_it/triple_it_2.wav",
        "data/voice_dataset/triple_it/triple_it_3.wav",
        "data/voice_dataset/triple_it/triple_it_4.wav",
        "data/voice_dataset/triple_it/triple_it_5.wav"
    ]
    
    print("Testing 'Triple IT' files:")
    for file_path in triple_it_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Dharwad" files
    dharwad_files = [
        "data/voice_dataset/dharwad/dharwad_1.wav",
        "data/voice_dataset/dharwad/dharwad_2.wav",
        "data/voice_dataset/dharwad/dharwad_3.wav",
        "data/voice_dataset/dharwad/dharwad_4.wav",
        "data/voice_dataset/dharwad/dharwad_5.wav"
    ]
    
    print("Testing 'Dharwad' files:")
    for file_path in dharwad_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Go" files
    go_files = [
        "data/voice_dataset/go/go_1.wav",
        "data/voice_dataset/go/go_2.wav",
        "data/voice_dataset/go/go_3.wav",
        "data/voice_dataset/go/go_4.wav",
        "data/voice_dataset/go/go_5.wav"
    ]
    
    print("Testing 'Go' files:")
    for file_path in go_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Hubli" files
    hubli_files = [
        "data/voice_dataset/hubli/hubli_1.wav",
        "data/voice_dataset/hubli/hubli_2.wav",
        "data/voice_dataset/hubli/hubli_3.wav",
        "data/voice_dataset/hubli/hubli_4.wav",
        "data/voice_dataset/hubli/hubli_5.wav"
    ]
    
    print("Testing 'Hubli' files:")
    for file_path in hubli_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Left" files
    left_files = [
        "data/voice_dataset/left/left_1.wav",
        "data/voice_dataset/left/left_2.wav",
        "data/voice_dataset/left/left_3.wav",
        "data/voice_dataset/left/left_4.wav",
        "data/voice_dataset/left/left_5.wav"
    ]
    
    print("Testing 'Left' files:")
    for file_path in left_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Right" files
    right_files = [
        "data/voice_dataset/right/right_1.wav",
        "data/voice_dataset/right/right_2.wav",
        "data/voice_dataset/right/right_3.wav",
        "data/voice_dataset/right/right_4.wav",
        "data/voice_dataset/right/right_5.wav"
    ]
    
    print("Testing 'Right' files:")
    for file_path in right_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")
    
    # Test on "Stop" files
    stop_files = [
        "data/voice_dataset/stop/stop_1.wav",
        "data/voice_dataset/stop/stop_2.wav",
        "data/voice_dataset/stop/stop_3.wav",
        "data/voice_dataset/stop/stop_4.wav",
        "data/voice_dataset/stop/stop_5.wav"
    ]
    
    print("Testing 'Stop' files:")
    for file_path in stop_files:
        predicted_word = predict_word(model, label_encoder, file_path)
        print(f"File: {file_path}, Predicted Word: {predicted_word}")