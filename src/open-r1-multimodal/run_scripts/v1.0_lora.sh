cd src/open-r1-multimodal
CUDA_VISIBLE_DEVICES=0,1
export DEBUG_MODE="true"
export WANDB_API_KEY=597622c60547b5e27fc630707414ef3ec6688986
RUN_NAME="Qwen2.5-VL-7B-GRPO-v2.1-loco-focalloss-v1-lora"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12350" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name loco \
    --data_file_paths data_jsonl/train_r1_loco_v2.1.jsonl \
    --image_folders /gpfsdata/home/yizhou/Project/AnomalyDetection/Datasets/mvtec_loco_anomaly_detection \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --reward_method yes_no \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 400 \
    --save_only_model false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true
