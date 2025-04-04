import os
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device.index if device.type == "cuda" else -1,
    return_timestamps=True,
    generate_kwargs={"language": "vi", "task": "transcribe"}
)

input_csv = 'multimed/Test/Vietnamese.csv'
output_csv = 'ASR/PhoWhisper-large/Test/Vietnamese.csv'

if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    existing_paths = set(existing_df['path'])
else:
    existing_paths = set()

df = pd.read_csv(input_csv)
paths = df['path']

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

new_entries = []
for path in tqdm(paths, desc="Transcribing"):
    if path in existing_paths:
        continue

    try:
        result = asr_pipeline(path)
        transcript = result['text']
        new_entries.append({'path': path, 'text': transcript})
        existing_paths.add(path)
    except Exception as e:
        print(f"Error processing {path}: {e}")

if new_entries:
    new_df = pd.DataFrame(new_entries)
    if os.path.exists(output_csv):
        new_df.to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
    else:
        new_df.to_csv(output_csv, mode='w', header=True, index=False, encoding='utf-8')
