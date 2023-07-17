#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python scripts/infer/infer_replace.py \
--ddim_steps 100 \
--ddim_eta 1 \
--outdir results/infer/replace_new_512 \
--config configs/replace.yaml \
--ckpt logs/replace/2023-07-17T08-38-48_replace/checkpoints/last.ckpt \
--seed 42 \
--reference /home/user01/data/polyp/new_kvasir/train_10/reference \
--dataset_path /home/user01/data/polyp/new_kvasir \
--H 512 \
--W 512 \
--n_samples 4 \
--scale 5