# import numpy as np
# import scipy.io.wavfile as wav
# from hmmlearn import hmm

# # Step 1: Load and preprocess audio
# def load_audio(file_path):
#     sample_rate, audio_data = wav.read(file_path)
#     audio_data = audio_data.astype(np.float32)  # Convert to float for processing
#     audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize
#     return sample_rate, audio_data

# # Step 2: Feature extraction (FFT-based)
# def compute_fft(signal):
#     n = len(signal)
#     fft_result = np.fft.fft(signal)
#     fft_magnitude = np.abs(fft_result)  # Use magnitude for comparison
#     return fft_magnitude

# def extract_features(audio_data, frame_size=256, overlap=128):
#     features = []
#     for i in range(0, len(audio_data) - frame_size, overlap):
#         frame = audio_data[i:i + frame_size]
#         fft_magnitude = compute_fft(frame)
#         features.append(fft_magnitude[:frame_size // 2])  # Use only the first half (symmetry)
#     return np.vstack(features)  # Convert to 2D array

# # Step 3: Train HMM for each word
# def train_hmm(features, n_components=3):
#     model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
#     model.fit(features)
#     return model

# # Step 4: Recognize a word using trained HMMs
# def recognize_word(audio_file, hmm_models):
#     sample_rate, audio_data = load_audio(audio_file)
#     features = extract_features(audio_data)

#     best_score = -float('inf')
#     best_word = None

#     for word, model in hmm_models.items():
#         score = model.score(features)  # Compute log-likelihood
#         if score > best_score:
#             best_score = score
#             best_word = word

#     return best_word

# # Example usage
# if __name__ == "__main__":
#     # Load training data
#     sample_rate, up_data = load_audio("data/train_dataset/up/Up-1.wav")
#     sample_rate, down_data = load_audio("data/train_dataset/down/Down-1.wav")

#     # Extract features
#     up_features = extract_features(up_data)
#     down_features = extract_features(down_data)

#     # Train HMMs
#     up_hmm = train_hmm(up_features)
#     down_hmm = train_hmm(down_features)

#     # Create a dictionary of HMM models
#     hmm_models = {
#         "up": up_hmm,
#         "down": down_hmm
#     }

#     # Recognize a word
#     audio_file = "data/train_dataset/up/Up-2.wav"
#     recognized_word = recognize_word(audio_file, hmm_models)
#     print(f"Recognized word: {recognized_word}")




# import numpy as np
# import scipy.io.wavfile as wav
# from scipy.fftpack import dct
# from hmmlearn import hmm

# # Step 1: Load and preprocess audio
# def load_audio(file_path):
#     """
#     Load an audio file and normalize it.
    
#     Parameters:
#     file_path: Path to the audio file
    
#     Returns:
#     sample_rate: Sample rate of the audio
#     audio_data: Normalized audio data (converted to mono if stereo)
#     """
#     sample_rate, audio_data = wav.read(file_path)
#     audio_data = audio_data.astype(np.float32)  # Convert to float for processing
    
#     # Convert stereo to mono by averaging channels if needed
#     if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
#         audio_data = np.mean(audio_data, axis=1)
    
#     # Normalize
#     audio_data = audio_data / np.max(np.abs(audio_data))
    
#     return sample_rate, audio_data

# # Step 2: Feature extraction using MFCC
# def compute_mfcc(signal, sample_rate, num_cepstral=13, frame_size=256, num_filters=26):
#     """
#     Compute MFCC features for a given audio signal.
    
#     Parameters:
#     signal: Audio signal (must be mono/1D)
#     sample_rate: Sample rate of audio
#     num_cepstral: Number of cepstral coefficients to return
#     frame_size: Size of each frame
#     num_filters: Number of mel-filters to use
    
#     Returns:
#     MFCC coefficients
#     """
#     # Ensure signal is 1D (mono)
#     if len(signal.shape) > 1:
#         raise ValueError("MFCC computation requires a mono signal")
        
#     # Make sure the frame has the correct length
#     if len(signal) < frame_size:
#         # Zero-pad if necessary
#         signal = np.pad(signal, (0, frame_size - len(signal)), 'constant')
    
#     # Apply Hamming window to reduce spectral leakage
#     hamming_window = np.hamming(len(signal))
#     signal_windowed = signal * hamming_window
    
#     # Compute FFT and get magnitude spectrum
#     fft_result = np.fft.rfft(signal_windowed)
#     fft_magnitude = np.abs(fft_result)
    
#     # Convert Hz to Mel scale
#     def hz_to_mel(hz):
#         return 2595 * np.log10(1 + hz/700)
    
#     def mel_to_hz(mel):
#         return 700 * (10**(mel/2595) - 1)
    
#     # Frequency range in Hz
#     low_freq = 0
#     high_freq = sample_rate / 2
    
#     # Convert to Mel scale
#     low_mel = hz_to_mel(low_freq)
#     high_mel = hz_to_mel(high_freq)
    
#     # Create equally spaced points in Mel scale
#     mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
#     # Convert back to Hz
#     hz_points = mel_to_hz(mel_points)
    
#     # Convert to FFT bin indices
#     bin_indices = np.floor((frame_size + 1) * hz_points / sample_rate).astype(int)
    
#     # Create filterbank
#     filterbank = np.zeros((num_filters, len(fft_magnitude)))
    
#     for i in range(num_filters):
#         # Ensure we don't exceed array bounds
#         lower_bound = min(bin_indices[i], len(fft_magnitude)-1)
#         middle_bound = min(bin_indices[i+1], len(fft_magnitude)-1)
#         upper_bound = min(bin_indices[i+2], len(fft_magnitude)-1)
        
#         # Create triangular filters
#         for j in range(lower_bound, middle_bound):
#             filterbank[i, j] = (j - lower_bound) / max(1, (middle_bound - lower_bound))
#         for j in range(middle_bound, upper_bound):
#             filterbank[i, j] = (upper_bound - j) / max(1, (upper_bound - middle_bound))
    
#     # Apply filterbank to spectrum
#     filtered_spectrum = np.dot(filterbank, fft_magnitude)
    
#     # Take logarithm
#     log_spectrum = np.log(filtered_spectrum + 1e-10)  # Add small constant to avoid log(0)
    
#     # Apply DCT
#     mfcc = dct(log_spectrum, type=2, norm='ortho')[:num_cepstral]
    
#     return mfcc

# def extract_mfcc_features(audio_data, sample_rate, frame_size=256, overlap=128, num_cepstral=13):
#     """
#     Extract MFCC features from audio data with frame-by-frame processing.
    
#     Parameters:
#     audio_data: Audio signal (mono)
#     sample_rate: Sample rate of audio
#     frame_size: Size of each frame
#     overlap: Overlap between frames
#     num_cepstral: Number of cepstral coefficients to return
    
#     Returns:
#     2D array of MFCC features, where each row contains the MFCC for a frame
#     """
#     features = []
    
#     # Print some debug info
#     print(f"Audio data shape: {audio_data.shape}")
#     print(f"Audio data type: {audio_data.dtype}")
    
#     for i in range(0, len(audio_data) - frame_size, overlap):
#         frame = audio_data[i:i + frame_size]
#         try:
#             mfcc_features = compute_mfcc(frame, sample_rate, num_cepstral, frame_size)
#             features.append(mfcc_features)
#         except Exception as e:
#             print(f"Error computing MFCC at frame {i}: {e}")
#             continue
    
#     # Make sure we have at least one feature vector
#     if len(features) == 0:
#         print("Warning: No features were successfully extracted")
#         return np.array([]).reshape(0, num_cepstral)  # Return empty array with correct shape
    
#     return np.vstack(features)  # Convert to 2D array

# # Step 3: Train HMM for each word
# def train_hmm(features, n_components=3):
#     """
#     Train a Hidden Markov Model on the given features.
    
#     Parameters:
#     features: Feature vectors (e.g., MFCC coefficients)
#     n_components: Number of hidden states in the HMM
    
#     Returns:
#     Trained HMM model
#     """
#     # Handle empty feature set or other error conditions
#     if features.size == 0:
#         raise ValueError("Empty feature set provided for training")
    
#     print(f"Training HMM with features of shape: {features.shape}")
    
#     model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
#     model.fit(features)
#     return model

# # Step 4: Recognize a word using trained HMMs
# def recognize_word(audio_file, hmm_models):
#     """
#     Recognize a word from an audio file using trained HMM models.
    
#     Parameters:
#     audio_file: Path to the audio file
#     hmm_models: Dictionary mapping words to their trained HMM models
    
#     Returns:
#     The recognized word
#     """
#     try:
#         sample_rate, audio_data = load_audio(audio_file)
#         features = extract_mfcc_features(audio_data, sample_rate)
        
#         if features.size == 0:
#             print("Warning: No features extracted from audio file")
#             return None
        
#         best_score = -float('inf')
#         best_word = None
        
#         for word, model in hmm_models.items():
#             try:
#                 score = model.score(features)  # Compute log-likelihood
#                 print(f"Score for '{word}': {score}")
#                 if score > best_score:
#                     best_score = score
#                     best_word = word
#             except Exception as e:
#                 print(f"Error scoring word '{word}': {e}")
        
#         return best_word
#     except Exception as e:
#         print(f"Error recognizing word: {e}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     try:
#         print("Loading training data...")
        
#         # Load training data
#         sample_rate, up_data = load_audio("data/train_dataset/up/Up-1.wav")
#         sample_rate, down_data = load_audio("data/train_dataset/down/Down-1.wav")
        
#         print(f"Up data shape: {up_data.shape}, Sample rate: {sample_rate}")
#         print(f"Down data shape: {down_data.shape}, Sample rate: {sample_rate}")
        
#         print("Extracting MFCC features...")
        
#         # Extract MFCC features
#         up_features = extract_mfcc_features(up_data, sample_rate)
#         down_features = extract_mfcc_features(down_data, sample_rate)
        
#         print(f"Up features shape: {up_features.shape}")
#         print(f"Down features shape: {down_features.shape}")
        
#         print("Training HMM models...")
        
#         # Train HMMs
#         up_hmm = train_hmm(up_features)
#         down_hmm = train_hmm(down_features)
        
#         # Create a dictionary of HMM models
#         hmm_models = {
#             "up": up_hmm,
#             "down": down_hmm
#         }
        
#         print("Recognizing test word...")
        
#         # Recognize a word
#         audio_file = "data/train_dataset/up/Up-2.wav"
#         recognized_word = recognize_word(audio_file, hmm_models)
#         print(f"Recognized word: {recognized_word}")
        
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()

import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from hmmlearn import hmm
import os

# Step 1: Load and preprocess audio
def load_audio(file_path):
    """
    Load an audio file and normalize it.
    
    Parameters:
    file_path: Path to the audio file
    
    Returns:
    sample_rate: Sample rate of the audio
    audio_data: Normalized audio data (converted to mono if stereo)
    """
    sample_rate, audio_data = wav.read(file_path)
    audio_data = audio_data.astype(np.float32)  # Convert to float for processing
    
    # Convert stereo to mono by averaging channels if needed
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return sample_rate, audio_data

# Step 2: Feature extraction using MFCC
def compute_mfcc(signal, sample_rate, num_cepstral=13, frame_size=256, num_filters=26):
    """
    Compute MFCC features for a given audio signal.
    
    Parameters:
    signal: Audio signal (must be mono/1D)
    sample_rate: Sample rate of audio
    num_cepstral: Number of cepstral coefficients to return
    frame_size: Size of each frame
    num_filters: Number of mel-filters to use
    
    Returns:
    MFCC coefficients
    """
    # Ensure signal is 1D (mono)
    if len(signal.shape) > 1:
        raise ValueError("MFCC computation requires a mono signal")
        
    # Make sure the frame has the correct length
    if len(signal) < frame_size:
        # Zero-pad if necessary
        signal = np.pad(signal, (0, frame_size - len(signal)), 'constant')
    
    # Apply Hamming window to reduce spectral leakage
    hamming_window = np.hamming(len(signal))
    signal_windowed = signal * hamming_window
    
    # Compute FFT and get magnitude spectrum
    fft_result = np.fft.rfft(signal_windowed)
    fft_magnitude = np.abs(fft_result)
    
    # Convert Hz to Mel scale
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz/700)
    
    def mel_to_hz(mel):
        return 700 * (10**(mel/2595) - 1)
    
    # Frequency range in Hz
    low_freq = 0
    high_freq = sample_rate / 2
    
    # Convert to Mel scale
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    
    # Create equally spaced points in Mel scale
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
    # Convert back to Hz
    hz_points = mel_to_hz(mel_points)
    
    # Convert to FFT bin indices
    bin_indices = np.floor((frame_size + 1) * hz_points / sample_rate).astype(int)
    
    # Create filterbank
    filterbank = np.zeros((num_filters, len(fft_magnitude)))
    
    for i in range(num_filters):
        # Ensure we don't exceed array bounds
        lower_bound = min(bin_indices[i], len(fft_magnitude)-1)
        middle_bound = min(bin_indices[i+1], len(fft_magnitude)-1)
        upper_bound = min(bin_indices[i+2], len(fft_magnitude)-1)
        
        # Create triangular filters
        for j in range(lower_bound, middle_bound):
            filterbank[i, j] = (j - lower_bound) / max(1, (middle_bound - lower_bound))
        for j in range(middle_bound, upper_bound):
            filterbank[i, j] = (upper_bound - j) / max(1, (upper_bound - middle_bound))
    
    # Apply filterbank to spectrum
    filtered_spectrum = np.dot(filterbank, fft_magnitude)
    
    # Take logarithm
    log_spectrum = np.log(filtered_spectrum + 1e-10)  # Add small constant to avoid log(0)
    
    # Apply DCT
    mfcc = dct(log_spectrum, type=2, norm='ortho')[:num_cepstral]
    
    return mfcc

def extract_mfcc_features(audio_data, sample_rate, frame_size=256, overlap=128, num_cepstral=13):
    """
    Extract MFCC features from audio data with frame-by-frame processing.
    
    Parameters:
    audio_data: Audio signal (mono)
    sample_rate: Sample rate of audio
    frame_size: Size of each frame
    overlap: Overlap between frames
    num_cepstral: Number of cepstral coefficients to return
    
    Returns:
    2D array of MFCC features, where each row contains the MFCC for a frame
    """
    features = []
    
    # Print some debug info
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Audio data type: {audio_data.dtype}")
    
    for i in range(0, len(audio_data) - frame_size, overlap):
        frame = audio_data[i:i + frame_size]
        try:
            mfcc_features = compute_mfcc(frame, sample_rate, num_cepstral, frame_size)
            features.append(mfcc_features)
        except Exception as e:
            print(f"Error computing MFCC at frame {i}: {e}")
            continue
    
    # Make sure we have at least one feature vector
    if len(features) == 0:
        print("Warning: No features were successfully extracted")
        return np.array([]).reshape(0, num_cepstral)  # Return empty array with correct shape
    
    return np.vstack(features)  # Convert to 2D array

# Step 3: Train HMM for each word
def train_hmm(features, n_components=3):
    """
    Train a Hidden Markov Model on the given features.
    
    Parameters:
    features: Feature vectors (e.g., MFCC coefficients)
    n_components: Number of hidden states in the HMM
    
    Returns:
    Trained HMM model
    """
    # Handle empty feature set or other error conditions
    if features.size == 0:
        raise ValueError("Empty feature set provided for training")
    
    print(f"Training HMM with features of shape: {features.shape}")
    
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    model.fit(features)
    return model

# Helper function to load and process all training examples for a word
def load_word_training_data(word_dir):
    """
    Load all training examples for a specific word.
    
    Parameters:
    word_dir: Directory containing audio files for the word
    
    Returns:
    List of feature arrays for each training example
    """
    all_features = []
    
    # Check if directory exists
    if not os.path.exists(word_dir):
        print(f"Warning: Directory {word_dir} does not exist")
        return []
    
    # Get all .wav files in the directory
    wav_files = [f for f in os.listdir(word_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"Warning: No .wav files found in {word_dir}")
        return []
    
    print(f"Loading {len(wav_files)} examples for word from {word_dir}")
    
    for wav_file in wav_files:
        try:
            file_path = os.path.join(word_dir, wav_file)
            sample_rate, audio_data = load_audio(file_path)
            features = extract_mfcc_features(audio_data, sample_rate)
            
            if features.size > 0:
                all_features.append(features)
            else:
                print(f"Warning: No features extracted from {file_path}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    return all_features

# Step 4: Recognize a word using trained HMMs
def recognize_word(audio_file, hmm_models):
    """
    Recognize a word from an audio file using trained HMM models.
    
    Parameters:
    audio_file: Path to the audio file
    hmm_models: Dictionary mapping words to their trained HMM models
    
    Returns:
    The recognized word
    """
    try:
        sample_rate, audio_data = load_audio(audio_file)
        features = extract_mfcc_features(audio_data, sample_rate)
        
        if features.size == 0:
            print("Warning: No features extracted from audio file")
            return None
        
        best_score = -float('inf')
        best_word = None
        
        for word, model in hmm_models.items():
            try:
                score = model.score(features)  # Compute log-likelihood
                print(f"Score for '{word}': {score}")
                if score > best_score:
                    best_score = score
                    best_word = word
            except Exception as e:
                print(f"Error scoring word '{word}': {e}")
        
        return best_word
    except Exception as e:
        print(f"Error recognizing word: {e}")
        return None

# Example usage
if __name__ == "__main__":
    try:
        # Define the 7 words to recognize
        words = ["up", "down", "left", "right", "yes", "no", "stop"]
        
        # Base directory for training data
        base_dir = "data/train_dataset"
        
        # Dictionary to store trained models
        hmm_models = {}
        
        print("Loading and processing training data for all words...")
        
        # Process each word
        for word in words:
            print(f"\nProcessing word: {word}")
            word_dir = os.path.join(base_dir, word)
            
            # Load all training examples for this word
            word_features_list = load_word_training_data(word_dir)
            
            if not word_features_list:
                print(f"Warning: No valid training data for word '{word}', skipping...")
                continue
            
            # Combine all training examples (if multiple exist)
            if len(word_features_list) > 1:
                combined_features = np.vstack(word_features_list)
                print(f"Combined {len(word_features_list)} examples, shape: {combined_features.shape}")
            else:
                combined_features = word_features_list[0]
                print(f"Using single example, shape: {combined_features.shape}")
            
            # Train HMM for this word
            try:
                word_hmm = train_hmm(combined_features)
                hmm_models[word] = word_hmm
                print(f"Successfully trained HMM for '{word}'")
            except Exception as e:
                print(f"Error training HMM for word '{word}': {e}")
                continue
        
        print("\nTraining complete. Number of trained models:", len(hmm_models))
        print("Trained words:", list(hmm_models.keys()))
        
        # Test recognition on a sample file
        if hmm_models:
            print("\nTesting recognition...")
            # You can modify this to test with different words
            test_word = ""  # Change this to test different words
            test_file = os.path.join(base_dir, test_word, f"{test_word.capitalize()}-2.wav")
            
            if os.path.exists(test_file):
                print(f"Testing with file: {test_file}")
                recognized_word = recognize_word(test_file, hmm_models)
                print(f"Ground truth: {test_word}")
                print(f"Recognized as: {recognized_word}")
                print(f"Correct: {recognized_word == test_word}")
            else:
                print(f"Test file {test_file} does not exist")
        else:
            print("No models were trained successfully. Cannot test recognition.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()