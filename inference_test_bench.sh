python scripts/inference_test_bench_new.py \
--ddim_steps 100 \
--ddim_eta 1 \
--outdir results/infer/Kvasir-SEG_10_high \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-28T09-56-41_polyp/checkpoints/last.ckpt \
--seed 42 \
--reference /home/majiajian/dataset/polyp/Kvasir-SEG/train_10/reference \
--dataset_path /home/majiajian/dataset/polyp/Kvasir-SEG \
--scale 5