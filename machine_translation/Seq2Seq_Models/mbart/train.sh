export CUDA_VISIBLE_DEVICES=0
python3 train.py \
    --model_name "facebook/mbart-large-50" \
    --source_language "Vietnamese" \
    --target_language "French" \
    --source_lang_symbol 'vi' \
    --target_lang_symbol 'fr' \
    --train_strategy "epoch" \
    --learning_rate 1e-5 \
    --warmup 50 \
    --train_batchsize 32 \
    --eval_batchsize 32 \
    --num_epochs 20 \
    --dataset "wnkh/MultiMed" \
    --max_input_length 128 \