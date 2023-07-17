#!/bin/bash
export CUDA_VISIBLE_DEVICES=2  # 指定使用 GPU 设备编号为 0


echo "$(date) - Starting" >> ./nohup.log

python scripts/infer/infer_inpaint.py \
--ddim_steps 100 \
--ddim_eta 1 \
--outdir results/infer/inpaint_updated_256 \
--config configs/replace.yaml \
--ckpt logs/replace/2023-07-17T08-38-48_replace/checkpoints/last.ckpt \
--seed 42 \
--reference /home/user01/data/polyp/new_kvasir/train_10_old/reference \
--dataset_path /home/user01/data/polyp/new_kvasir \
--normal_dir /home/user01/data/polyp/normal-cecum \
--H 256 \
--W 256 \
--n_samples 10 \
--scale 5

echo "$(date) - Finishing" >> ./nohup.log