from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
import torch
import pandas as pd
import os
import re
import string
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Authenticate
login(token="HF Token")

# Load evaluation metrics
metrics = {
    'bleu': evaluate.load("sacrebleu"),
    'ter': evaluate.load("ter"),
    'meteor': evaluate.load("meteor"),
    'rouge': evaluate.load("rouge"),
    'chrf': evaluate.load("chrf")
}

def compute_metrics(predictions, references):
    """Compute evaluation metrics for predictions."""
    return {
        metric: (m.compute(predictions=predictions, references=[[r] for r in references]) if metric != 'rouge' else m.compute(predictions=predictions, references=references))
        for metric, m in metrics.items()
    }

def generate_translations(source_texts, model, tokenizer, batch_size=16):
    """Generate translations from model."""
    translations = []
    for i in range(0, len(source_texts), batch_size):
        batch = source_texts[i:i+batch_size]
        model_inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**model_inputs)
        translations.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    return translations

def normalize_text_to_chars(text):
    """Normalize text by removing punctuation and spacing out characters."""
    if not isinstance(text, str):
        return text
    return " ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", text).strip())

def prepare_data(dataset, source_column, target_column):
    """Prepare dataset, handling missing values."""
    dataset.fillna("", inplace=True)
    return dataset[source_column].tolist(), dataset[target_column].tolist()

def evaluate_dataset(src_texts, ref_texts, model, tokenizer, target_language):
    """Evaluate dataset with model and tokenizer."""
    translations = generate_translations(src_texts, model, tokenizer)
    if target_language == 'Chinese':
        translations = [normalize_text_to_chars(t) for t in translations]
    return translations, compute_metrics(translations, ref_texts)

def save_results(translations, references, metric_scores, source_language, target_language, model_name):
    """Save translations and metric results to CSV files."""
    output_file = f"marian_{source_language}_{target_language}_translations.csv"
    pd.DataFrame({'Translation': translations, 'Reference': references}).to_csv(output_file, index=False)
    
    metrics_file = 'translation_metrics.csv'
    metrics_df = pd.DataFrame([{ 'Model': f'{model_name}_{source_language}_{target_language}', **{m.upper(): s for m, s in metric_scores.items()} }])
    if os.path.exists(metrics_file):
        metrics_df = pd.concat([pd.read_csv(metrics_file), metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Results saved to {output_file} and {metrics_file}")

languages = ["English", "Vietnamese", "French", "German", "Chinese"]
language_map = {'English': 'en', 'Vietnamese': 'vi', 'French': 'fr', 'German': 'de', 'Chinese': 'zh'}
language_sym = {'English': 'en_XX', 'Vietnamese': 'vi_VN', 'French': 'fr_XX', 'German': 'de_DE', 'Chinese': 'zh_CN'}

df = pd.read_csv("translation_metrics.csv")
inferred_models = set(df['Model'])

source_language = 'English'
for target_language in languages:
    if source_language == target_language:
        continue
    
    print(f"Processing {target_language}...")
    folder_path = f'marian-finetuned-{language_map[source_language]}-to-{language_map[target_language]}'
    checkpoints = sorted([d for d in os.listdir(folder_path) if d.startswith("checkpoint-") and d.split('-')[-1].isdigit()], key=lambda x: int(x.split('-')[-1]), reverse=True)
    
    if not checkpoints:
        print(f"No checkpoints found for {source_language} to {target_language}. Skipping...")
        continue
    
    model_name = f"{folder_path}/{checkpoints[0]}"
    if f'{model_name}_{source_language}_{target_language}' in inferred_models:
        print(f"Skipping {source_language} to {target_language}, already evaluated.")
        continue
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=language_sym[source_language], tat_lang=language_sym[target_language])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    dataset = pd.read_csv(f'{target_language}.csv')
    src_texts, ref_texts = prepare_data(dataset, source_column='Prediction', target_column=target_language)
    
    if target_language == 'Chinese':
        ref_texts = [normalize_text_to_chars(t) for t in ref_texts]
    
    translations, results = evaluate_dataset(src_texts, ref_texts, model, tokenizer, target_language)
    save_results(translations, ref_texts, results, source_language, target_language, model_name)
