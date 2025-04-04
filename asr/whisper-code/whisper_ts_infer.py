import os
import re
import string
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Device and Data Type Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Language Mapping
language_map = {
    'English': 'en',
    'Vietnamese': 'vi',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh'
}

source = 'Chinese'
target = 'Vietnamese'
lang1 = language_map[source]
lang2 = language_map[target]

# Prepare Checkpoint Directory
temp_ckpt_folder = 'whisper/temp'
ckpt_dir = f'whisper_ts/whisper-{lang1}-{lang2}/checkpoint'

os.makedirs(temp_ckpt_folder, exist_ok=True)
ckpt_dir_parent = str(Path(ckpt_dir).parent)

files_to_copy = [
    "added_tokens.json", "normalizer.json", "preprocessor_config.json",
    "special_tokens_map.json", "generation_config.json", "tokenizer_config.json",
    "merges.txt", "vocab.json", "config.json", "model.safetensors", "training_args.bin"
]
os.system(f"cp {' '.join([f'{ckpt_dir_parent}/{file}' for file in files_to_copy])} {temp_ckpt_folder}")
model_id = temp_ckpt_folder

# Load Model and Processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id, language=lang2, task="translate")
forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang1, task="translate")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
)

# Load Dataset
dataset = load_dataset('wnkh/MultiMed', source, split='corrected.test')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ASR Inference
def transcribe_audio(batch):
    """Transcribes audio using the Whisper model."""
    batch["prediction"] = pipe(batch["audio"], return_timestamps=True)['text']
    return batch

# Run Inference and Save Results
def infer_and_save_to_csv(dataset, lang1, lang2, target):
    """Processes the dataset, transcribes audio, and saves predictions to a CSV file."""
    predictions = []
    references = []
    
    for batch in tqdm(dataset, desc="Transcribing"):
        transcript = pipe(batch["audio"], return_timestamps=True)['text']
        predictions.append(transcript)
        references.append(batch[target])
    
    df = pd.DataFrame({
        "Reference": references,
        "Translation": predictions
    })
    
    output_file = f"TS_result/{lang1}_{lang2}.csv"
    df.to_csv(output_file, index=False)
    print(f"Predictions and corresponding texts saved to {output_file}")
    
infer_and_save_to_csv(dataset, lang1, lang2, target)
