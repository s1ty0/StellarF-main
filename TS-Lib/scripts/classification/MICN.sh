export CUDA_VISIBLE_DEVICES=0

model_name=MICN

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset_k \
  --model_id tyx_MICN \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --c_out 3 \
  --patience 10

