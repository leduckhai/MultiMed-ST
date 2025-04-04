import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
import re
import string
import numpy as np
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from huggingface_hub import login

HF_TOKEN = "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"
login(token=HF_TOKEN)

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
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
    help='Language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--target_language', 
    type=str, 
    required=False, 
    default='Vietnamese', 
    help='Language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--source_sym', 
    type=str, 
    required=True, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--target_sym', 
    type=str, 
    required=True, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=2, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
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
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default="", 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default="", 
    help='List of datasets to be used for evaluation.'
)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_lower_case = False
do_remove_punctuation = False
# punctuation_to_remove = string.punctuation.replace("'", "")
punctuation_to_remove_regex = f"[{re.escape(string.punctuation)}]"

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.target_language, task="translate")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.target_language, task="translate")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
model.generation_config.language = args.source_language
model.generation_config.task = "translate"
# model.config.apply_spec_augment = True
# model.config.mask_time_prob = 0.05
# model.config.mask_feature_prob = 0.05

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False

if gradient_checkpointing:
    model.config.use_cache = False

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_length"] = len(batch["audio"])
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # optional pre-processing steps
    transcription = batch[args.target_language]
    if batch[args.target_language] == None:
        batch[args.target_language] = ''
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = re.sub(punctuation_to_remove_regex, " ", transcription).strip()
    batch["labels"] = tokenizer(batch[args.target_language], truncation=True, max_length=448).input_ids
    batch["labels_length"] = len(tokenizer(batch[args.target_language], add_special_tokens=False).input_ids)
    return batch

def filter_labels(labels_length):
    return labels_length < 448
    
print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = load_dataset("wnkh/MultiMed", args.source_language)
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
raw_dataset = raw_dataset.map(prepare_dataset, remove_columns=raw_dataset.column_names["train"], num_proc=args.num_proc)

raw_dataset = raw_dataset.filter(filter_labels, num_proc=1, input_columns=['input_length'])
raw_dataset = raw_dataset.filter(filter_labels, num_proc=1, input_columns=['labels_length'])

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("sacrebleu")    
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

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=False,
        resume_from_checkpoint=args.resume_from_ckpt,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=False,
        resume_from_checkpoint=args.resume_from_ckpt,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()
print('DONE TRAINING')