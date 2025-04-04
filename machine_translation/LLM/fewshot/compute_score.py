import os
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import re
import string

# Set up evaluation metrics
metrics = {
    'bleu': evaluate.load("sacrebleu"),
    'ter': evaluate.load("ter"),
    'meteor': evaluate.load("meteor"),
    'rouge': evaluate.load("rouge"),
    'chrf': evaluate.load("chrf")
}

def normalize_text_to_chars(text):
    punctuation_to_remove_regex = f"[{re.escape(string.punctuation)}]"
    # Replace punctuation with a space
    text = re.sub(punctuation_to_remove_regex, " ", text).strip() if isinstance(text, str) else text
    # Make each character a word by inserting spaces
    return " ".join(text) if isinstance(text, str) else text

def compute_score(predict_translations, reference_translations, target_lang):
    # if target_lang == 'Chinese':
    #     reference_translations = [normalize_text_to_chars(sentence) for sentence in reference_translations]
    #     predict_translations = [normalize_text_to_chars(sentence) for sentence in predict_translations]
    results = {}
    for metric_name, metric in metrics.items():
        if metric_name == 'rouge':
            results[metric_name] = metric.compute(
                predictions=predict_translations,
                references=reference_translations
            )
        else:
            results[metric_name] = metric.compute(
                predictions=predict_translations,
                references=[[ref] for ref in reference_translations]
            )
    
    return results

def save_scores_to_csv(model_name, metric_scores, output_file='/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/asr-fewshot/eval_zh_de.csv'):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame()

    new_row = {'Model': f'{model_name}_{source_language}_{target_language}'}
    new_row.update({metric.upper(): score for metric, score in metric_scores.items()})

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(output_file, index=False)

# Languages
languages = ["Chinese", 'Vietnamese', 'French', 'German']
csv_file = "/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/asr-fewshot/eval_zh_de.csv"
df = pd.read_csv(csv_file)
model_infer = list(df['Model'])
source_language = 'Chinese'
models = ['qwenaudio2']
domains = ['gt', 'mono', 'multi', 'openai', 'assembly', 'deepgram']
shot = [0, 1, 8, 16, 32]
# name_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Loop through source and target language combinations
for model in models:
    for num in shot:
        # for target_language in languages:
        target_language = 'German'
        # try:
        #path = f'/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/fewshot/{model}/{model}_{source_language}_{target_language}_{num}_shot.csv'
        path = '/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/Qwenaudio/result_new/QwenAudio2_Chinese_German.csv'
        df = pd.read_csv(path)
        if len(df) > 96:
            df = df.iloc[0:96]
        metric_scores = compute_score(
            df['Translation'], df['Reference'],target_language
        )

        # Save scores
        save_scores_to_csv(f'{model}_ft', metric_scores)

        # except:
        #     path = f'/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/codeswitch/gt/{model}_{source_language}_{target_language}.csv'
        #     df = pd.read_csv(path)
            
        #     metric_scores = compute_score(
        #         df['Translation'], df['Reference'],target_language
        #     )

        #     # Save scores
        #     save_scores_to_csv(model, metric_scores)