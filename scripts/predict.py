# import os
# import librosa
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder

# # Function to extract MFCC features
# def extract_mfcc(file_path, max_pad_len=100):
#     audio, sr = librosa.load(file_path, sr=None)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     pad_width = max_pad_len - mfccs.shape[1]
#     if pad_width > 0:
#         mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfccs = mfccs[:, :max_pad_len]
#     return mfccs

# # Function to load the trained model and label encoder
# def load_model_and_encoder(model_path, label_encoder_path):
#     # Load the trained Keras model
#     model = tf.keras.models.load_model(model_path)
    
#     # Load the label encoder classes
#     label_encoder_classes = np.load(label_encoder_path, allow_pickle=True)
#     label_encoder = LabelEncoder()
#     label_encoder.classes_ = label_encoder_classes
    
#     return model, label_encoder

# # Function to detect keywords in a given audio file
# def detect_keywords_in_sentence(model, label_encoder, file_path, keywords, window_size=100, stride=50):
#     # Check if the file exists
#     if not os.path.exists(file_path):
#         print(f"Error: File not found at {file_path}")
#         return {}
    
#     try:
#         # Load the audio file
#         audio, sr = librosa.load(file_path, sr=None)
        
#         # Extract MFCC features for the entire audio
#         mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
#         # Normalize features (use the same normalization as during training)
#         mfccs = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)
        
#         # Dictionary to store detection results
#         detection_results = {keyword: "no" for keyword in keywords}
        
#         # Use a sliding window to analyze segments of the audio
#         for i in range(0, mfccs.shape[1] - window_size, stride):
#             segment = mfccs[:, i:i + window_size]
            
#             # Reshape for model input (add batch dimension)
#             segment = np.expand_dims(segment, axis=0)  # Shape: (1, num_frames, num_mfcc_coefficients)
            
#             # Predict
#             predictions = model.predict(segment)
#             predicted_class_index = np.argmax(predictions, axis=1)[0]
#             predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            
#             # Check if the predicted class matches any of the keywords
#             if predicted_class in detection_results:
#                 detection_results[predicted_class] = "yes"
        
#         return detection_results
#     except Exception as e:
#         print(f"Error processing audio file {file_path}: {e}")
#         return {}

# # Main script
# if __name__ == "__main__":
#     # Paths to the trained model and label encoder
#     model_path = "models/kws_model.keras"  # Path to the trained Keras model
#     label_encoder_path = "models/label_encoder_classes.npy"  # Path to the label encoder classes
    
#     # Load the model and label encoder
#     model, label_encoder = load_model_and_encoder(model_path, label_encoder_path)
    
#     # Path to the folder containing test audio files
#     test_audio_folder = "data/test_audio"  # Replace with your test audio folder path
    
#     # Check if the test audio folder exists
#     if not os.path.exists(test_audio_folder):
#         print(f"Error: Test audio folder not found at {test_audio_folder}")
#         exit(1)
    
#     # List of keywords to detect (from the training dataset)
#     keywords = ["dharwad", "down", "go", "hackmaze", "hubli", "left", "right", "stop", "triple IT", "up"]
    
#     # Iterate through all files in the test audio folder
#     for file_name in os.listdir(test_audio_folder):
#         # Skip hidden files/folders
#         if file_name.startswith('.'):
#             continue
        
#         # Construct the full file path
#         file_path = os.path.join(test_audio_folder, file_name)
        
#         # Detect keywords in the current audio file
#         detection_results = detect_keywords_in_sentence(model, label_encoder, file_path, keywords)
        
#         # Print detection results for the current file
#         print(f"\nResults for file: {file_name}")
#         for keyword, result in detection_results.items():
#             print(f"{keyword}: {result}")