CUDA_VISIBLE_DEVICES=0 python main.py \
--train \
--train_batch_size 16 \
--eval_batch_size 16 \
--checkpoint_dir "/data/hurunyi/MLTC/checkpoints/Roberta/best" \
--load_from_checkpoint
