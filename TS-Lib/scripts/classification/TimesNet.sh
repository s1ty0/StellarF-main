export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset_k \
  --model_id tyx_TimesNet \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10
