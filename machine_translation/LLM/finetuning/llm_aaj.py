import os
import pandas as pd
import glob
import re
from tqdm import tqdm
import time
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="API Key")

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 10,
    "max_output_tokens": 2048,  
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

save_path = 'llm_aag_evaluation.csv'

def process_row(row, source_lang, target_lang):
    """Process a single row with Gemini"""
    
    evaluation_prompt = f"""
        Please act as an impartial translation quality assessment expert. Evaluate the Assistant's Translation by analyzing these aspects:

        1. **Correctness** (Faithfulness to source):
        - Compare with Source Text ({source_lang})
        - Check for omissions/additions/distortions
        - Preservation of semantic meaning

        2. **Fluency** (Target language quality):
        - Naturalness in {target_lang}
        - Grammatical correctness
        - Idiomatic expression

        3. **Terminology** (Domain consistency):
        - Specialized term consistency
        - Comparison with Reference Translation
        - Proper noun handling

        **Source Text ({source_lang}):**
        {row['source sentence']}

        **Reference Translation ({target_lang}):**
        {row['reference sentence']}

        **Assistant's Translation ({target_lang}):**
        {row['prediction sentence']}

        **Evaluation Steps:**
        1. Analyze errors in each category
        2. Classify error severity (minor/major/critical)
        3. Provide specific examples of errors
        4. Assign numerical score (1-10 scale):
        - 1-4: Poor (meaning distorted)
        - 5-6: Fair (meaning preserved but with issues)
        - 7-8: Good (minor errors)
        - 9-10: Excellent (near-perfect)

        Format your response as:
        **Errors:**
        - [Category] Description (severity)
        Example: "Original: X | Translation: Y"

        **Rating:**
        [[N]] (Score between 1-10)

        Provide only the formatted response starting with **Errors:**"""

    response = model.generate_content(evaluation_prompt)
    response_text = response.text.strip()
    
    # Extract the score using regex
    SCORE_PATTERN = re.compile(r"\[\[(\d+)\]\]")
    match = SCORE_PATTERN.search(response_text)
    return int(match.group(1)) if match else -1

def process_with_retry(row, source_lang, target_lang, max_retries=3):
    """Retry function to handle temporary failures"""
    for attempt in range(max_retries):
        try:
            score = process_row(row, source_lang, target_lang)
            return score
        except Exception as e:
            wait_time = (2 ** attempt) * 5  
            print(f"Error: {e}. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    return -1

def load_existing_results():
    """Load existing results from CSV file"""
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    return pd.DataFrame(columns=['Model', 'Score'])

def save_result(name, avg_score, existing_df):
    """Save a single result to CSV"""
    new_row = pd.DataFrame({'Model': [name], 'Score': [avg_score]})
    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    updated_df.to_csv(save_path, index=False)
    return updated_df

results_df = load_existing_results()
processed_models = results_df['Model'].tolist()

for file_path in tqdm(glob.glob('/home/tdtuyen/wespeaker/wespeaker/speaker_models/nmt/LLM/asr_eval/qwen/*'), desc="Processing folders"):
    name = os.path.splitext(os.path.basename(file_path))[0]
    
    if name in processed_models:
        continue
        
    df = pd.read_excel(file_path)
    source_lang, target_lang = name.split('_')[1], name.split('_')[2]
    
    scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {name}"):
        score = process_with_retry(row, source_lang, target_lang)
        if score != -1:
            tqdm.write(f"Score: {score}")
            scores.append(score)
    
    if scores:
        avg_score = round(sum(scores) / len(scores), 2)
        results_df = save_result(name, avg_score, results_df)
        print(f"Saved results for {name} with average score: {avg_score}")
    else:
        print(f"No valid scores for {name}")

print(f"All evaluation results saved to {save_path}")