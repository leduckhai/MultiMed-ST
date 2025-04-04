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

MODEL_NAME="Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
TRAIN_STRATEGY="epoch"
LEARNING_RATE=2e-5
WARMUP=225
TRAIN_BATCHSIZE=16
EVAL_BATCHSIZE=16
NUM_EPOCHS=10
DATASET="wnkh/MultiMed"
MAX_INPUT_LENGTH=256

for SRC_LANG in "${LANGUAGES[@]}"; do
    for TGT_LANG in "${LANGUAGES[@]}"; do
        if [ "$SRC_LANG" == "$TGT_LANG" ]; then
            continue
        fi

        SRC_SYM=${LANGUAGE_CODES[$SRC_LANG]}
        TGT_SYM=${LANGUAGE_CODES[$TGT_LANG]}

        OUTPUT_DIR="result/marian-${SRC_SYM}-${TGT_SYM}" 
        
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Output directory $OUTPUT_DIR already exists. Skipping training for $SRC_LANG -> $TGT_LANG."
            continue
        fi
        
        echo "Training for source: $SRC_LANG ($SRC_SYM) -> target: $TGT_LANG ($TGT_SYM)"
        
        python3 process_data.py \
            --model_name "$MODEL_NAME" \
            --source_language "$SRC_LANG" \
            --target_language "$TGT_LANG" \
            --source_lang_symbol "$SRC_SYM" \
            --target_lang_symbol "$TGT_SYM" \
            --train_strategy "$TRAIN_STRATEGY" \
            --learning_rate $LEARNING_RATE \
            --warmup $WARMUP \
            --train_batchsize $TRAIN_BATCHSIZE \
            --eval_batchsize $EVAL_BATCHSIZE \
            --num_epochs $NUM_EPOCHS \
            --dataset "$DATASET" \
            --max_input_length $MAX_INPUT_LENGTH \
            --output_dir "$OUTPUT_DIR"
            
        echo "Finished training for source: $SRC_LANG ($SRC_SYM) -> target: $TGT_LANG ($TGT_SYM)"
    done
done
