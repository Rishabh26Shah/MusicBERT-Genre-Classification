import os
import json
import random
from collections import Counter

# --- CONFIG ---
subset = "topmagd"
json_path = "midi_genre_map.json"
midi_dir = "topmagd_data_raw/0/midis"
output_dir = "processed"
label_dict_path = os.path.join(output_dir, "label")
input0_dir = os.path.join(output_dir, "input0")
CROP_LENGTH = 1024  # Must be a multiple of 8

# --- SETUP OUTPUT ---
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dict_path, exist_ok=True)
os.makedirs(input0_dir, exist_ok=True)

# --- LOAD GENRE MAPPING ---
with open(json_path) as f:
    full_map = json.load(f)
genre_map = full_map.get(subset, {})

# --- GET MIDI FILES + LABELS ---
file_label_pairs = []
for midi_file in os.listdir(midi_dir):
    if midi_file.endswith(".mid"):
        midi_id = os.path.splitext(midi_file)[0]
        if midi_id in genre_map:
            label = genre_map[midi_id]
            if isinstance(label, list):
                label = label[0]
            path = os.path.join(midi_dir, midi_file)
            file_label_pairs.append((path, label))

# --- SHUFFLE & SPLIT ---
random.shuffle(file_label_pairs)
n = len(file_label_pairs)
splits = {
    "train": file_label_pairs[:int(n * 0.8)],
    "valid": file_label_pairs[int(n * 0.8):int(n * 0.9)],
    "test": file_label_pairs[int(n * 0.9):]
}

# --- GENRE INDEXING ---
all_genres = sorted(set(label for _, label in file_label_pairs))
label_to_idx = {genre: i for i, genre in enumerate(all_genres)}
label_counter = Counter(label for _, label in file_label_pairs)

# --- WRITE TEXT, LABEL, INPUT0 FILES ---
for split, data in splits.items():
    txt_path = os.path.join(output_dir, f"midi_{split}.txt")
    lbl_path = os.path.join(output_dir, f"midi_{split}.label")
    input_path = os.path.join(input0_dir, f"{split}.input0")

    with open(txt_path, "w") as f_txt, open(lbl_path, "w") as f_lbl, open(input_path, "w") as f_inp:
        for path, label in data:
            f_txt.write(f"{path}\n")
            f_lbl.write(f"{label_to_idx[label]}\n")
            f_inp.write(f"{path}\n")

# --- WRITE LABEL DICT.TXT ---
with open(os.path.join(label_dict_path, "dict.txt"), "w") as f:
    for genre in all_genres:
        f.write(f"{genre} {label_counter[genre]}\n")

# --- REPORT ---
print(f"✅ Created train/valid/test sets with {len(splits['train'])} / {len(splits['valid'])} / {len(splits['test'])} entries")
print(f"🎵 Genres (index mapping): {label_to_idx}")
print(f"📐 crop_length used: {CROP_LENGTH} (must be multiple of 8)")

# --- OPTIONAL: CHECK MULTIPLE OF 8 (simulated check, placeholder) ---
if CROP_LENGTH % 8 != 0:
    print("❌ ERROR: CROP_LENGTH is not a multiple of 8!")
else:
    print("✅ All sequences will be generated with lengths multiple of 8.")
