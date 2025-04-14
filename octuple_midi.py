import pretty_midi
import numpy as np
import os

OCTUPLE_FIELDS = [
    'Bar',          # bar position
    'Position',     # beat position within bar
    'Pitch',        # note pitch (0–127)
    'Duration',     # duration of note
    'Velocity',     # velocity of note
    'Program',      # instrument program (0–127)
    'Tempo',        # tempo class
    'Chord'         # chord class (or placeholder)
]

def encode_midi(midi_path, crop_length=None):
    """
    Encodes a MIDI file into the OctupleMIDI format.
    
    Args:
        midi_path (str): Path to the MIDI file.
        crop_length (int): Maximum number of octuples (tokens) to return.
        
    Returns:
        str: A whitespace-separated sequence of tokens (flattened octuples).
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[WARN] Failed to parse MIDI: {midi_path} -- {e}")
        return None

    tokens = []
    bar_length = 4.0  # assume 4/4 time
    resolution = 32   # grid resolution within bar
    max_duration = 4.0  # seconds
    chord_placeholder = 0  # you could add real chord detection here

    for instrument in midi.instruments:
        program = instrument.program
        for note in instrument.notes:
            bar = int(note.start // bar_length)
            pos = int((note.start % bar_length) / bar_length * resolution)
            pitch = note.pitch
            dur = min(note.end - note.start, max_duration)
            dur_class = int(dur * 10)  # simple quantization
            vel_class = int(note.velocity / 16)  # 0–127 → 0–7
            tempo = int(midi.get_tempo_changes()[1][0] // 10)  # quantize tempo
            tokens.append([
                bar,
                pos,
                pitch,
                dur_class,
                vel_class,
                program,
                tempo,
                chord_placeholder
            ])

    # Sort by bar/pos for proper ordering
    tokens.sort(key=lambda x: (x[0], x[1]))

    # Flatten to a sequence of tokens
    flattened = []
    for t in tokens:
        flattened.extend(map(str, t))

    if crop_length:
        flattened = flattened[:crop_length * len(OCTUPLE_FIELDS)]

    return " ".join(flattened)
