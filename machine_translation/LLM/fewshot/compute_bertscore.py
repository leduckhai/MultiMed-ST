import os
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import re
import string
import glob
import numpy as np

language = {
    'English': 'en',
    'Vietnamese': 'vi',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh'
}

bert_score = evaluate.load('bertscore')

def score(translations, reference_translations, lang):
    results = {}
    result = bert_score.compute(
        predictions=translations,
        references=[[ref] for ref in reference_translations],
        lang=lang
    )
    P, R, F1 = result['precision'], result['recall'], result['f1']
    Precision = float(np.mean(P))
    Recall = float(np.mean(R))
    F1_score = float(np.mean(F1))
    results['Precision'] = Precision
    results['Recall'] = Recall
    results['F1'] = F1_score
    print(results)
    return results

def save_scores_to_csv(model_name, metric_scores, output_file='/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/asr-fewshot/bert_score_zh_de.csv'):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        data = {
            'Model': [],  
            'PRECISION': [],  
            'RECALL': [],  
            'F1': []  
        }
        df = pd.DataFrame(data)

    new_row = {'Model': f'{model_name}'}
    new_row.update({metric.upper(): score for metric, score in metric_scores.items()})

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(output_file, index=False)
    
source_language = 'Chinese'
models = ['qwenaudio2']
shot = [0, 1, 8, 16, 32]
domains = ['gt', 'mono', 'multi', 'openai', 'assembly', 'deepgram']
    
#for i in glob.glob('/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/LLM/qwen-asr/*.csv'):
for model in models:
    for num in shot:
        # for target_language in languages:
        target_language = 'German'
        # try:
        #path = f'/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/fewshot/{model}/{model}_{source_language}_{target_language}_{num}_shot.csv'
        path = '/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/Qwenaudio/result_new/QwenAudio2_Chinese_German.csv'
        # lang = i.split('_')[1]
        lang = language[target_language]
        df = pd.read_csv(path)
        if len(df) > 96:
            df = df.iloc[0:96]
        df.fillna("", inplace=True)
        
        prediction = df['Translation']
        reference = df['Reference']
        
        result = score(prediction, reference, lang=lang)
        save_scores_to_csv(f'{model}_ft', result)
        break 
    break