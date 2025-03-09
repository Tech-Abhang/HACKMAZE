import numpy as np
import librosa
from hmmlearn import hmm
import os

# Parameters
N_STATES = 5      # Number of states per HMM
N_MFCC = 13       # Number of MFCC features
N_KEYWORDS = 10   # Number of keywords

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16 kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCCs
    return mfcc.T  # Transpose to get (num_frames x n_mfcc)

# Function to train an HMM for a keyword
def train_hmm_for_keyword(mfcc_sequences, n_states=5):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    lengths = [len(seq) for seq in mfcc_sequences]  # Lengths of each sequence
    mfcc_data = np.vstack(mfcc_sequences)  # Stack all sequences into a single array
    model.fit(mfcc_data, lengths)  # Train the HMM
    return model

# Function to train HMMs for all keywords
def train_hmms(keyword_files, n_states=5):
    hmms = []
    keyword_order = []  # To keep track of the order of keywords
    
    for keyword in keyword_files.keys():
        mfcc_sequences = [extract_mfcc(file) for file in keyword_files[keyword]]
        hmm_model = train_hmm_for_keyword(mfcc_sequences, n_states)
        hmms.append(hmm_model)
        keyword_order.append(keyword)
        
    return hmms, keyword_order

# Function to export HMM parameters to fixed-point format
def export_hmm_params(hmm_model, fractional_bits=16):
    def to_fixed_point(value):
        return int(value * (1 << fractional_bits))

    # Export transition matrix
    transition_matrix = np.vectorize(to_fixed_point)(hmm_model.transmat_)

    # Export emission means and variances
    emission_means = np.vectorize(to_fixed_point)(hmm_model.means_)
    emission_variances = np.vectorize(to_fixed_point)(hmm_model.covars_)

    return transition_matrix, emission_means, emission_variances

# Dictionary of keywords and their corresponding audio files
keyword_files = {
    "dharwad": [
        "data/train_dataset/dharwad/Dharwad_0.wav", "data/train_dataset/dharwad/Dharwad-1.wav",
        "data/train_dataset/dharwad/Dharwad-2.wav", "data/train_dataset/dharwad/Dharwad-3.wav",
        "data/train_dataset/dharwad/Dharwad.wav"
    ],
    "down": [
        "data/train_dataset/down/Down-1.wav", "data/train_dataset/down/Down-2.wav",
        "data/train_dataset/down/Down-3.wav", "data/train_dataset/down/Down.wav"
    ],
    "go": [
        "data/train_dataset/go/go_0.wav", "data/train_dataset/go/Go-1.wav",
        "data/train_dataset/go/Go-2.wav", "data/train_dataset/go/Go-3.wav",
        "data/train_dataset/go/Go.wav"
    ],
    "hackmaze": [
        "data/train_dataset/hackmaze/Hack Maze 3.wav", "data/train_dataset/hackmaze/Hack Maze_0.wav",
        "data/train_dataset/hackmaze/Hack Maze.wav", "data/train_dataset/hackmaze/Hackmaze-1.wav",
        "data/train_dataset/hackmaze/Hackmaze-2.wav"
    ],
    "hubli": [
        "data/train_dataset/hubli/Hubli_0.wav", "data/train_dataset/hubli/Hubli-1.wav",
        "data/train_dataset/hubli/Hubli-2.wav", "data/train_dataset/hubli/Hubli-3.wav",
        "data/train_dataset/hubli/Hubli.wav"
    ],
    "left": [
        "data/train_dataset/left/left_0.wav", "data/train_dataset/left/Left-1.wav",
        "data/train_dataset/left/Left-2.wav", "data/train_dataset/left/Left-3.wav",
        "data/train_dataset/left/Left.wav"
    ],
    "right": [
        "data/train_dataset/right/right_0.wav", "data/train_dataset/right/Right-1.wav",
        "data/train_dataset/right/Right-2.wav", "data/train_dataset/right/Right-3.wav",
        "data/train_dataset/right/Right.wav"
    ],
    "stop": [
        "data/train_dataset/stop/stop_0.wav", "data/train_dataset/stop/Stop-1.wav",
        "data/train_dataset/stop/Stop-2.wav", "data/train_dataset/stop/Stop-3.wav",
        "data/train_dataset/stop/Stop.wav"
    ],
    "triple_it": [
        "data/train_dataset/triple_it/Triple IT_0.wav", "data/train_dataset/triple_it/Triple IT-1.wav",
        "data/train_dataset/triple_it/Triple IT-2.wav", "data/train_dataset/triple_it/Triple IT-3.wav",
        "data/train_dataset/triple_it/Triple IT.wav"
    ],
    "up": [
        "data/train_dataset/up/up_0.wav", "data/train_dataset/up/Up-1.wav",
        "data/train_dataset/up/Up-2.wav", "data/train_dataset/up/Up-3.wav",
        "data/train_dataset/up/Up.wav"
    ]
}

# Create output directory for memory files
os.makedirs("mem_files", exist_ok=True)

# Train HMMs for all keywords
hmms, keyword_order = train_hmms(keyword_files, n_states=N_STATES)

# Export HMM parameters
hmm_params = {}
for keyword, hmm_model in zip(keyword_order, hmms):
    transition_matrix, emission_means, emission_variances = export_hmm_params(hmm_model)
    hmm_params[keyword] = (transition_matrix, emission_means, emission_variances)

# Create memory initialization files for $readmemh
# 1. Transition matrix file
with open("mem_files/transition_matrix.mem", "w") as f:
    for keyword in keyword_order:
        trans_mat = hmm_params[keyword][0]
        for i in range(N_STATES):
            for j in range(N_STATES):
                f.write(f"{int(trans_mat[i, j]):08x}\n")  # Convert to integer

# 2. Emission means file
with open("mem_files/emission_mean.mem", "w") as f:
    for keyword in keyword_order:
        means = hmm_params[keyword][1]
        for i in range(N_STATES):
            for j in range(N_MFCC):
                f.write(f"{int(means[i, j]):08x}\n")  # Convert to integer

# 3. Emission variances file
with open("mem_files/emission_var.mem", "w") as f:
    for keyword in keyword_order:
        vars_matrix = hmm_params[keyword][2]
        for i in range(N_STATES):
            for j in range(N_MFCC):
                value = vars_matrix[i, j]
                
                # If value is an array, extract its first element
                if isinstance(value, np.ndarray):
                    value = value[0]
                
                f.write(f"{int(value):08x}\n")  # Convert to integer

print("Memory files generated in 'mem_files' directory")