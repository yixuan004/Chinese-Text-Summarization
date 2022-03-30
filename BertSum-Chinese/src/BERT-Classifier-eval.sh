export CUDA_VISIBLE_DEVICES=2,3

python train_LAI.py \
-mode test \
-bert_data_path ../bert_data/LCSTS \
-model_path ../models/bert_classifier \
-visible_gpus 1 \
-gpu_ranks 0 \
-batch_size 30000 \
-log_file tmp.log \
-result_path ../results/LCSTS \
-test_all \
-block_trigram False \
-test_from ../models/bert_classifier/model_step_10000.pt \
-report_rouge False