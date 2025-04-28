#!/usr/bin/env python3

import os
import csv

# Set these paths
WAV_DIR = "./FurbSounds"            
OUTPUT_CSV = "./metadata.csv" 

def clean_text_from_filename(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Replace underscores with spaces
    text = name.replace('_', ' ')
    return text

def generate_metadata_from_filenames(wav_dir, output_csv):
    wav_files = sorted(f for f in os.listdir(wav_dir) if f.lower().endswith('.wav'))
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        for wav_file in wav_files:
            wav_id = os.path.splitext(wav_file)[0]  # e.g., hey_me_see_you
            transcript = clean_text_from_filename(wav_file)  # "hey me see you"
            writer.writerow([wav_id, transcript])

    print(f"✅ Generated metadata.csv with {len(wav_files)} entries from filenames.")

if __name__ == "__main__":
    generate_metadata_from_filenames(WAV_DIR, OUTPUT_CSV)
