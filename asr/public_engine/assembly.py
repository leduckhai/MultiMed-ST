import os
import json
import pandas as pd
from tqdm import tqdm
import assemblyai as aai

def load_existing_paths(csv_label):
    """Load existing paths from the CSV file to avoid redundant transcription."""
    if os.path.exists(csv_label):
        existing_df = pd.read_csv(csv_label)
        return set(existing_df['path'])
    return set()

def transcribe_audio(path, transcriber):
    """Transcribe an audio file using AssemblyAI's API."""
    try:
        transcript = transcriber.transcribe(path)
        return transcript.text
    except Exception as e:
        print(f"Error transcribing {path}: {e}")
        return None

def main():
    csv_input = 'multimed/Test/Vietnamese.csv'
    csv_output = 'Assembly/Test/Vietnamese.csv'
    
    aai.settings.api_key = "API key"
    config = aai.TranscriptionConfig(language_code="vi")
    transcriber = aai.Transcriber(config=config)
    
    df = pd.read_csv(csv_input)
    existing_paths = load_existing_paths(csv_output)
    new_entries = []

    for path in tqdm(df['path'], desc="Transcribing"):
        if path in existing_paths:
            continue

        transcript = transcribe_audio(path, transcriber)
        if transcript:
            new_entries.append([path, transcript])
            existing_paths.add(path)
    
    if new_entries:
        pd.DataFrame(new_entries, columns=['path', 'text']).to_csv(
            csv_output, mode='a', header=not os.path.exists(csv_output), index=False, encoding='utf-8'
        )

if __name__ == "__main__":
    main()
