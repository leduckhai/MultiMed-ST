from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, 
                          M2M100ForConditionalGeneration, M2M100Tokenizer)
import evaluate
import torch
import pandas as pd
import os
import string
import re

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load evaluation metrics
metrics = {
    'bleu': evaluate.load("sacrebleu"),
    'ter': evaluate.load("ter"),
    'meteor': evaluate.load("meteor"),
    'rouge': evaluate.load("rouge"),
    'chrf': evaluate.load("chrf")
}

# Compute evaluation metrics
def compute_metrics(predictions, references):
    results = {}
    for metric_name, metric in metrics.items():
        if metric_name == 'rouge':
            results[metric_name] = metric.compute(predictions=predictions, references=references)
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return results

# Normalize text for Chinese translation
def normalize_text_to_chars(text):
    if not isinstance(text, str):
        return text
    return " ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", text).strip())

# Prepare data for translation
def prepare_data(dataset, source_column, target_column):
    dataset.fillna("", inplace=True)
    return dataset[source_column].tolist(), dataset[target_column].tolist()

# Generate translations
def generate_translations(source_texts, model, tokenizer, batch_size=16):
    translations = []
    for i in range(0, len(source_texts), batch_size):
        batch = source_texts[i:i + batch_size]
        model_inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**model_inputs)
        translations.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    return translations

# Save results
def save_results(translations, reference_translations, metric_scores, source_lang, target_lang, model_name):
    output_dir = "output"
    output_file = f"{output_dir}/m2m100_{source_lang}_{target_lang}_translations.csv"
    metrics_file = f"{output_dir}/translation_metrics.csv"
    
    pd.DataFrame({'Translation': translations, 'Reference': reference_translations}).to_csv(output_file, index=False)
    
    metrics_df = pd.DataFrame([{'Model': f'{model_name}_{source_lang}_{target_lang}',
                                **{metric.upper(): score for metric, score in metric_scores.items()}}])
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Results saved to {output_file} and {metrics_file}")

# Translation evaluation pipeline
def evaluate_dataset(dataset, model, tokenizer, source_column, target_column):
    src_texts, ref_texts = prepare_data(dataset, source_column, target_column)
    if target_column == 'Chinese':
        ref_texts = [normalize_text_to_chars(text) for text in ref_texts]
    translations = generate_translations(src_texts, model, tokenizer)
    if target_column == 'Chinese':
        translations = [normalize_text_to_chars(text) for text in translations]
    return translations, compute_metrics(translations, ref_texts)

# Language mappings
languages = ["English", "Vietnamese", "French", "German", "Chinese"]
language_codes = {'English': 'en', 'Vietnamese': 'vi', 'French': 'fr', 'German': 'de', 'Chinese': 'zh'}
language_syms = {'English': 'en_XX', 'Vietnamese': 'vi_VN', 'French': 'fr_XX', 'German': 'de_DE', 'Chinese': 'zh_CN'}

# Load existing model inference records
metrics_csv = "translation_metrics.csv"
inferred_models = pd.read_csv(metrics_csv)['Model'].tolist() if os.path.exists(metrics_csv) else []

# Perform translation for each language pair
source_language = 'English'
for target_language in languages:
    if source_language == target_language:
        continue
    
    model_dir = f"m2m100_418M-finetuned-{language_codes[source_language]}-to-{language_codes[target_language]}"
    checkpoints = sorted([d for d in os.listdir(model_dir) if d.startswith("checkpoint-") and d.split('-')[-1].isdigit()])
    model_name = f"{model_dir}/{checkpoints[0]}" if checkpoints else None
    
    if not model_name or f'{model_name}_{source_language}_{target_language}' in inferred_models:
        continue
    
    # Load model and tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=language_codes[source_language], tat_lang=language_codes[target_language])
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Load dataset
    dataset_path = f"{target_language}.csv"
    dataset = pd.read_csv(dataset_path)
    
    # Evaluate translations
    translations, results = evaluate_dataset(dataset, model, tokenizer, source_column='Prediction', target_column=target_language)
    
    # Save results
    save_results(translations, dataset[target_language].tolist(), results, source_language, target_language, model_name)