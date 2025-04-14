import miditoolkit
import os

# Set path to one specific MIDI file from the Clean MIDI Dataset
midi_path = "/Users/Rishabh/Documents/Vanderbilt/Spring 2025/Transformers/musicbert/LMD_Clean/2_Brothers_on_the_4th_Floor/Dreams.mid"

print("Checking MIDI file:", midi_path)

# Check if the file exists
if not os.path.exists(midi_path):
    print("File not found:", midi_path)
else:
    try:
        # Load the MIDI file
        midi = miditoolkit.MidiFile(midi_path)

        # Count instruments and notes
        num_instruments = len(midi.instruments)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)

        print("File loaded successfully.")
        print("Instruments:", num_instruments)
        print("Total notes:", total_notes)

        if total_notes == 0:
            print("Warning: This file contains no notes and will be skipped in preprocessing.")

    except Exception as e:
        print("Failed to parse MIDI file:", e)
