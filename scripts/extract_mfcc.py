import librosa
import numpy as np

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

# Example usage
if __name__ == "__main__":
    file_path = "data/test_audio/sentence1.wav"
    mfccs = extract_mfcc(file_path)
    print("MFCC shape:", mfccs.shape)