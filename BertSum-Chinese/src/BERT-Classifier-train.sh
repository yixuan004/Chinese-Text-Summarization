export CUDA_VISIBLE_DEVICES=2,3

python train_LAI.py \
-mode train \
-encoder classifier \
-dropout 0.1 \
-bert_data_path ../bert_data/LCSTS \
-model_path ../models/bert_classifier-selfdata \
-lr 2e-3 \
-visible_gpus 1 \
-gpu_ranks 0 \
-world_size 1 \
-report_every 50 \
-save_checkpoint_steps 100 \
-batch_size 1000 \
-decay_method noam \
-train_steps 30000 \
-accum_count 2 \
-log_file ../logs/bert_classifier \
-use_interval true \
-warmup_steps 10000