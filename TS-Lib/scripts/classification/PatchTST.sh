export CUDA_VISIBLE_DEVICES=1

model_name=PatchTST

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset_k \
  --model_id tyx_PatchTST \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

