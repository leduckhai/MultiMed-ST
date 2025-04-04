import os
import json
import pandas as pd
from tqdm import tqdm
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

def load_existing_paths(csv_label):
    """Load existing paths from the CSV file to avoid redundant transcription."""
    if os.path.exists(csv_label):
        existing_df = pd.read_csv(csv_label)
        return set(existing_df['path'])
    return set()

def transcribe_audio(path, client):
    """Transcribe an audio file using Deepgram's API."""
    try:
        with open(path, "rb") as file:
            buffer_data = file.read()

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-2", smart_format=True, language='de', sample_rate=16000
        )

        response = client.listen.rest.v("1").transcribe_file(payload, options)
        data = json.loads(response.to_json())
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        print(f"Error transcribing {path}: {e}")
        return None

def main():
    csv_input = 'multimed/Eval/German.csv'
    csv_output = 'Deepgram/Eval/German.csv'
    deepgram = DeepgramClient('API Key')
    
    df = pd.read_csv(csv_input)
    existing_paths = load_existing_paths(csv_output)
    new_entries = []

    for path in tqdm(df['path'], desc="Transcribing"):
        if path in existing_paths:
            continue

        transcript = transcribe_audio(path, deepgram)
        if transcript:
            new_entries.append([path, transcript])
            existing_paths.add(path)
    
    if new_entries:
        pd.DataFrame(new_entries, columns=['path', 'text']).to_csv(
            csv_output, mode='a', header=not os.path.exists(csv_output), index=False, encoding='utf-8'
        )

if __name__ == "__main__":
    main()
