#!/bin/bash
echo "$(date) - Starting" >> ./nohup.log

python scripts/infer/infer_outpaint.py \
--ddim_steps 100 \
--ddim_eta 1 \
--outdir results/infer/outpaint_elastic \
--config configs/outpaint.yaml \
--ckpt logs/outpaint/2023-07-10T11-53-58_outpaint/checkpoints/last.ckpt \
--seed 42 \
--reference /home/user01/data/polyp/combine/train \
--dataset_path /home/user01/data/polyp/new_kvasir \
--H 256 \
--W 256 \
--k 4 \
--random_shift True \
--scale 5

echo "$(date) - Finishing" >> ./nohup.log