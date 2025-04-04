import os
import re
import math
import string
import torch
import pandas as pd
import spacy
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
import evaluate

# -------------------- Model Configuration -------------------- #
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)
model.eval()
torch.set_grad_enabled(False)

# Load evaluation metrics
METRICS = {
    'bleu': evaluate.load("sacrebleu"),
    'ter': evaluate.load("ter"),
    'meteor': evaluate.load("meteor"),
    'rouge': evaluate.load("rouge"),
    'chrf': evaluate.load("chrf")
}

# -------------------- Utility Functions -------------------- #
def normalize_text_to_chars(text):
    """Normalize text by removing punctuation and inserting spaces between characters."""
    if not isinstance(text, str):
        return text
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text).strip()
    return " ".join(text)

def extract_translation(text):
    """Extract the translated text from model output."""
    match = re.search(r'Response:\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_final_response(text):
    """Extract the final translation response in few-shot scenarios."""
    match = re.search(r"Now, translate this text:\s*Input:.*?\nResponse:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def zero_shot_prompt(input_text, instruction):
    return f"{instruction}\nInput: {input_text}\nResponse:"

def few_shot_prompt(input_text, examples, source_lang, target_lang, instruction):
    if target_lang == 'Chinese':
        examples = [{'input': ex['input'], 'output': normalize_text_to_chars(ex['output'])} for ex in examples]
    prompt = f"{instruction}\n\n" + "\n\n".join([f"Example {i+1}:\nInput: {ex['input']}\nResponse: {ex['output']}" for i, ex in enumerate(examples)])
    return f"{prompt}\n\nNow, translate this text:\nInput: {input_text}\nResponse:"

def load_examples_from_csv(csv_path, input_col='text', output_col=None):
    """Load few-shot examples from CSV."""
    df = pd.read_csv(csv_path)
    return [{'input': row[input_col], 'output': row[output_col]} for _, row in df.iterrows() if pd.notna(row[input_col]) and pd.notna(row[output_col])]

def translate_and_evaluate(source_sentences, reference_translations, src_lang, tgt_lang, instruction, batch_size, shot, examples):
    """Translate sentences and evaluate using metrics."""
    translations = []
    for i in tqdm(range(0, len(source_sentences), batch_size), desc="Translating"):
        batch = source_sentences[i:i+batch_size]
        prompts = [zero_shot_prompt(sent, instruction) if shot == '0_shot' else few_shot_prompt(sent, examples, src_lang, tgt_lang, instruction) for sent in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend([extract_translation(text) if shot == '0_shot' else extract_final_response(text) for text in decoded])
    
    results = {name: metric.compute(predictions=translations, references=[[ref] for ref in reference_translations]) for name, metric in METRICS.items()}
    return translations, results

def save_scores_to_csv(model_name, metric_scores, num, src_lang, tgt_lang, output_file):
    """Save evaluation scores to CSV."""
    df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()
    new_row = {"Model": f"{model_name}_{src_lang}_{tgt_lang}_{num}_shot", **{metric.upper(): score for metric, score in metric_scores.items()}}
    pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).to_csv(output_file, index=False)

# -------------------- Translation Pipeline -------------------- #
src_lang = 'English'
target_languages = ["Vietnamese", "French", "German", "Chinese"]
evaluation_file = "evaluation_scores.csv"
df = pd.read_csv(evaluation_file) if os.path.exists(evaluation_file) else pd.DataFrame()
already_processed = set(df['Model'])

for num_shot in [1]:
    for tgt_lang in target_languages:
        model_key = f"{MODEL_NAME}_{src_lang}_{tgt_lang}_{num_shot}_shot"
        if model_key in already_processed:
            continue
        
        print(f"Processing {src_lang} -> {tgt_lang} ({num_shot}-shot)...")
        dataset = load_dataset("wnkh/MultiMed", src_lang)
        source_sentences, reference_translations = dataset['corrected.test']['text'], dataset['corrected.test'][tgt_lang]
        source_sentences, reference_translations = zip(*[(s, r) for s, r in zip(source_sentences, reference_translations) if s and r])
        
        ##### If use ASR text input
        # df = pd.read_csv(f"{src_lang}_predictions.csv")
        
        # if tgt_lang not in df.columns:
        #     raise ValueError(f"Column for {tgt_lang} not found in the input CSV")
        
        # source_texts = df['Prediction'].fillna('').tolist()
        # reference_texts = df[tgt_lang].fillna('').tolist()
        
        if num_shot > 0:
            example_path = f"{src_lang}/sample_{num_shot}.csv"
            examples = load_examples_from_csv(example_path, output_col=tgt_lang)
        else:
            examples = None
        
        instruction = f"Translate the following sentence from {src_lang} to {tgt_lang}:"
        translations, metrics = translate_and_evaluate(source_sentences, reference_translations, src_lang, tgt_lang, instruction, batch_size=64 if num_shot <= 1 else 16, shot=f'{num_shot}_shot', examples=examples)
        
        output_file = f"qwen_{src_lang}_{tgt_lang}_{num_shot}_shot.csv"
        pd.DataFrame({'Translation': translations, 'Reference': reference_translations}).to_csv(output_file, index=False)
        save_scores_to_csv(MODEL_NAME, metrics, num_shot, src_lang, tgt_lang, evaluation_file)
        print(f"Saved translations to {output_file}.")
