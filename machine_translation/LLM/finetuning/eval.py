import os
import re
import string
import torch
import pandas as pd
import evaluate
import spacy
from tqdm import tqdm
from unsloth import FastLanguageModel

# === Model settings ===
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# Define prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Define evaluation metrics
METRICS = {
    'bleu': evaluate.load("sacrebleu"),
    'ter': evaluate.load("ter"),
    'meteor': evaluate.load("meteor"),
    'rouge': evaluate.load("rouge"),
    'chrf': evaluate.load("chrf")
}

# Load NLP model for sentence extraction
nlp = spacy.load("xx_ent_wiki_sm")
if 'sentencizer' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')

# === Utility Functions ===
def normalize_text(text):
    """Normalize text by removing punctuation and spacing out characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text).strip()
    return " ".join(text)

def extract_first_sentence(text):
    """Extract the first sentence using Spacy or the first non-empty line."""
    if not text or str(text).strip() == "":
        return ""
    
    if '\n' in text:
        return text.split('\n')[0].strip()
    
    return text.strip()

def extract_translation(text):
    """Extract generated translation from the response format."""
    match = re.search(r'### Response:\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def zero_shot_prompt(input_text, instruction):
    """Format input text with Alpaca-style instruction prompt."""
    return ALPACA_PROMPT.format(instruction, input_text, "")

def compute_metrics(predictions, references):
    """Compute BLEU, TER, METEOR, ROUGE, and CHRF scores."""
    results = {}
    for metric_name, metric in METRICS.items():
        if metric_name == 'rouge':
            results[metric_name] = metric.compute(predictions=predictions, references=references)
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return results

def save_results(model, translations, references, metric_scores, source_lang, target_lang, model_name):
    """Save translation results and evaluation metrics to CSV files."""
    output_dir = "multi_result"
    os.makedirs(output_dir, exist_ok=True)

    translation_file = f"{output_dir}/{model}_{source_lang}_{target_lang}_translations.csv"
    metrics_file = "multilingual_score.csv"

    # Save translations
    pd.DataFrame({'Translation': translations, 'Reference': references}).to_csv(translation_file, index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{'Model': f'{model_name}_{source_lang}_{target_lang}', **{metric.upper(): score for metric, score in metric_scores.items()}}])
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Results saved to {translation_file} and {metrics_file}")

# === Translation Function ===
def translate_and_evaluate(model, tokenizer, source_texts, reference_texts, source_lang, target_lang, instruction, batch_size=64):
    """Translate source texts and compute evaluation scores."""
    translations = []

    for i in tqdm(range(0, len(source_texts), batch_size), desc=f"Translating {source_lang} → {target_lang}"):
        batch_sentences = source_texts[i:i+batch_size]
        batch_prompts = [zero_shot_prompt(sentence, instruction) for sentence in batch_sentences]

        # Tokenize and generate translations
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract and normalize translations
        res = [extract_translation(text) for text in batch_translations]
        if target_lang == 'Chinese':
            res = [normalize_text(sentence) for sentence in res]
        res = [extract_first_sentence(text) for text in res]

        translations.extend(res)

    # Compute evaluation scores
    metric_scores = compute_metrics(translations, reference_texts)
    return translations, metric_scores

# === Main Function ===
def main(model_name, input_csv, source_lang, target_lang, model_path):
    """Load data, translate, evaluate, and save results."""
    df = pd.read_csv(input_csv)
    if target_lang not in df.columns:
        raise ValueError(f"Column for {target_lang} not found in the input CSV")

    source_texts = df['Prediction'].fillna("").tolist()
    reference_texts = df[target_lang].fillna("").tolist()

    # Normalize if necessary
    if source_lang == 'Chinese':
        source_texts = [normalize_text(text) for text in source_texts]
    if target_lang == 'Chinese':
        reference_texts = [normalize_text(text) for text in reference_texts]

    instruction = f"Translate the following sentence from {source_lang} to {target_lang}:"
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    FastLanguageModel.for_inference(model)
    model.eval()
    torch.set_grad_enabled(False)

    # Check if model inference already exists
    metric_file = "/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/LLM/multilingual_score.csv"
    if os.path.exists(metric_file):
        df_metrics = pd.read_csv(metric_file)
        if f'{model_path}_{source_lang}_{target_lang}' in df_metrics['Model'].values:
            print(f"Skipping {source_lang} → {target_lang}, already evaluated.")
            return

    # Translate and evaluate
    translations, metric_scores = translate_and_evaluate(
        model, tokenizer, source_texts, reference_texts, source_lang, target_lang, instruction, batch_size=8
    )

    # Save results
    save_results(model_name, translations, reference_texts, metric_scores, source_lang, target_lang, model_path)

    # Free GPU memory
    torch.cuda.empty_cache()

# === Execution ===
if __name__ == "__main__":
    source_languages = ['Vietnamese', 'French', 'German', 'Chinese', 'English']
    target_languages = ['Vietnamese', 'French', 'German', 'Chinese', 'English']
    models = ['mistral', 'llama', 'qwen']

    for src_lang in source_languages:
        input_csv = f"/Test/{src_lang}_predictions.csv"

        for model_name in models:
            model_path = f"multilingual/{model_name}_multilingual_translation/checkpoint"
            
            for tgt_lang in target_languages:
                if src_lang == tgt_lang:
                    continue
                
                print(f"Translating {src_lang} → {tgt_lang} using {model_name}")
                main(model_name, input_csv, src_lang, tgt_lang, model_path)
