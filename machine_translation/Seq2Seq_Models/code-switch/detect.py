import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def detect_codeswitching(sentence, language):
    """Detects code-switching between the given language and English."""
    try:
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Analyze the following sentence for code-switching between {language} and English. "
                           f"Respond with 'Yes' if code-switching is present, and 'No' if not. \n Sentence: {sentence}"
            }],
            model="llama-3.1-70b-versatile",
        )
        response_text = response.choices[0].message.content.strip().lower()

        return "Yes" if "yes" in response_text else "No" if "no" in response_text else f"Undetermined: {response_text}"

    except Exception as e:
        return f"Error: {str(e)}"

def filter_code_switching_sentences(dataset, language):
    """Filters sentences containing code-switching."""
    filtered_sentences = []

    for sample in tqdm(dataset, desc="Processing Sentences"):
        result = detect_codeswitching(sample.get('German', ''), language)
        print(result)

        if result == "Yes":
            filtered_sentences.append(sample)

    return filtered_sentences

def main():
    language = 'Deutsch'
    dataset = load_dataset("wnkh/MultiMed", 'English', split='corrected.test').remove_columns(['audio', 'duration'])
    
    filtered_sentences = filter_code_switching_sentences(dataset, language)

    if filtered_sentences:
        df = pd.DataFrame(filtered_sentences)
        output_path = f'{language}.csv'
        df.to_csv(output_path, index=False)
        print(f"Filtered dataset saved to: {output_path}")
    else:
        print("No code-switching sentences found.")

if __name__ == "__main__":
    main()
