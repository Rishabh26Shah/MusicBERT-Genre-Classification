import os
import random
from musicbert.octuple_midi import encode_midi

# --- CONFIG ---
MIDI_DIR = "topmagd_data_raw/0/midis"
OUTPUT_DIR = "processed/input0"
CROP_LENGTH = 1024  # must be a multiple of 8

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLLECT MIDI FILES ---
all_midi_files = [os.path.join(MIDI_DIR, f) for f in os.listdir(MIDI_DIR) if f.endswith(".mid")]
random.shuffle(all_midi_files)
n = len(all_midi_files)

train_files = all_midi_files[:int(n * TRAIN_RATIO)]
valid_files = all_midi_files[int(n * TRAIN_RATIO):int(n * (TRAIN_RATIO + VALID_RATIO))]
test_files = all_midi_files[int(n * (TRAIN_RATIO + VALID_RATIO)):] 

splits = {
    "train": train_files,
    "valid": valid_files,
    "test": test_files
}

# --- ENCODE AND WRITE ---
for split, files in splits.items():
    with open(os.path.join(OUTPUT_DIR, f"{split}.input0"), "w") as fout:
        for path in files:
            try:
                tokens = encode_midi(path)
                # Pad or truncate tokens to make sure slices are multiple of 8
                if len(tokens) < CROP_LENGTH:
                    continue  # skip short files
                for i in range(0, len(tokens) - CROP_LENGTH + 1, CROP_LENGTH):
                    chunk = tokens[i:i + CROP_LENGTH]
                    fout.write(" ".join(map(str, chunk)) + "\n")
            except Exception as e:
                print(f"⚠️ Skipping {path} due to error: {e}")

print("✅ Encoded all MIDI files to OctupleMIDI format with crop length", CROP_LENGTH) 
