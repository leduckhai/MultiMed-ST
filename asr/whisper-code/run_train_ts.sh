#!/bin/bash
declare -A LANGUAGE_CODES
LANGUAGE_CODES=(
    ["English"]="en"
    ["Vietnamese"]="vi"
    ["French"]="fr"
    ["German"]="de"
    ["Chinese"]="zh"
)

LANGUAGES=("English" "Vietnamese" "French" "German" "Chinese")

MODEL_NAME="openai/whisper-small"
SAMPLING_RATE=16000
NUM_PROC=2
TRAIN_STRATEGY="steps"
LEARNING_RATE=1e-5
WARMUP=500
TRAIN_BATCHSIZE=8
EVAL_BATCHSIZE=8
NUM_STEPS=4000
TRAIN_DATASETS="wnkh/MultiMed"
EVAL_DATASETS="wnkh/MultiMed"

for SRC_LANG in "${LANGUAGES[@]}"; do
    for TGT_LANG in "${LANGUAGES[@]}"; do
        if [ "$SRC_LANG" == "$TGT_LANG" ]; then
            continue
        fi

        SRC_SYM=${LANGUAGE_CODES[$SRC_LANG]}
        TGT_SYM=${LANGUAGE_CODES[$TGT_LANG]}

        OUTPUT_DIR="st/whisper-${SRC_SYM}-${TGT_SYM}" 
        
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Output directory $OUTPUT_DIR already exists. Skipping training for $SRC_LANG -> $TGT_LANG."
            continue
        fi
        
        echo "Training for source: $SRC_LANG ($SRC_SYM) -> target: $TGT_LANG ($TGT_SYM)"
        
        python3 train_ts.py \
            --model_name "$MODEL_NAME" \
            --source_language "$SRC_LANG" \
            --target_language "$TGT_LANG" \
            --source_sym "$SRC_SYM" \
            --target_sym "$TGT_SYM" \
            --sampling_rate $SAMPLING_RATE \
            --num_proc $NUM_PROC \
            --train_strategy "$TRAIN_STRATEGY" \
            --learning_rate $LEARNING_RATE \
            --warmup $WARMUP \
            --train_batchsize $TRAIN_BATCHSIZE \
            --eval_batchsize $EVAL_BATCHSIZE \
            --num_steps $NUM_STEPS \
            --output_dir "$OUTPUT_DIR" \
            --train_datasets "$TRAIN_DATASETS" \
            --eval_datasets "$EVAL_DATASETS"
            
        echo "Finished training for source: $SRC_LANG ($SRC_SYM) -> target: $TGT_LANG ($TGT_SYM)"
    done
done
