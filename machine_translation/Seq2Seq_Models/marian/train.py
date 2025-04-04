import warnings
import numpy as np

from datasets import load_dataset, Dataset, DatasetDict
import evaluate

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning script for Marian Models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--source_language', 
    type=str, 
    required=False, 
    default='Vietnamese', 
    help='Source language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--target_language', 
    type=str, 
    required=False, 
    default='Vietnamese', 
    help='Target language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--source_lang_symbol', 
    type=str, 
    required=False, 
    default='vi', 
    help='Source language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--target_lang_symbol', 
    type=str, 
    required=False, 
    default='zh_CN', 
    help='Target language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=48, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=20, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--dataset', 
    type=str, 
    required=True, 
    default="", 
    help='dataset to be used for training.'
)
parser.add_argument(
    '--max_input_length', 
    type=int,  
    required=False, 
    default=128, 
)
parser.add_argument(
    '--prefix', 
    type=str,  
    required=False, 
    default='', 
)
parser.add_argument(
    '--output_dir', 
    type=str,  
    required=False, 
    default='', 
)

args = parser.parse_args()

# Load datasets
data = DatasetDict()
data = load_dataset(args.dataset, args.source_language)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.src_lang = args.source_lang_symbol
tokenizer.tgt_lang = args.target_lang_symbol
# Initialize model and training arguments
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

def transform_for_translation(dataset, source_lang, target_lang, name_source_lang, name_target_lang):
    transformed_data = []
    
    for idx in range(len(dataset)):
        example = dataset[idx]  
        unique_id = str(idx)
        
        if example[name_source_lang] is None or example[name_target_lang] is None:
            example[name_source_lang] = ""
            example[name_target_lang] = "" 
                
        translation = {
            source_lang: example[name_source_lang],  
            target_lang: example[name_target_lang]   
        }

        transformed_data.append({
            'id': unique_id,
            'translation': translation
        })
    
    return Dataset.from_dict({
        'id': [item['id'] for item in transformed_data],
        'translation': [item['translation'] for item in transformed_data],
    })

def preprocess_function(examples):               
    inputs = [args.prefix + example[args.source_lang_symbol] for example in examples["translation"]]
    targets = [example[args.target_lang_symbol] for example in examples["translation"]]
    
    
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu = evaluate.load("sacrebleu")
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    return result

# Prepare data
data_train = transform_for_translation(data['train'], source_lang=args.source_lang_symbol, target_lang=args.target_lang_symbol, name_source_lang='text', name_target_lang=args.target_language)
data_eval = transform_for_translation(data['eval'], source_lang=args.source_lang_symbol, target_lang=args.target_lang_symbol, name_source_lang='text', name_target_lang=args.target_language)
data_test = transform_for_translation(data['corrected.test'], source_lang=args.source_lang_symbol, target_lang=args.target_lang_symbol, name_source_lang='text', name_target_lang=args.target_language)

dataset = DatasetDict({
    'train': data_train,
    'eval': data_eval,
    'test': data_test
})

data_final = dataset.map(preprocess_function, batched=True)

name_model = args.model_name.split("/")[-1]

model_args = Seq2SeqTrainingArguments(
    args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    learning_rate=args.learning_rate, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.03,
    label_smoothing_factor=0.1,
    eval_accumulation_steps=3,
    num_train_epochs=args.num_epochs,
    warmup_steps=args.warmup,
    save_total_limit=2, 
    predict_with_generate=True, 
    max_grad_norm=1.0,
)

# Create a data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=data_final['train'],
    eval_dataset=data_final['eval'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
