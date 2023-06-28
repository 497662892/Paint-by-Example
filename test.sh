python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-20T16-56-07_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1hhj6mxfp90835n3wofrap.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1hhj6mxfp90835n3wofrap.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju2qqn5ys4uo0988ewrt2ip2_ref_0.jpg \
--seed 321 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1

python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-20T16-56-07_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1hhj6mxfp90835n3wofrap.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1hhj6mxfp90835n3wofrap.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju1dfeupuzlw0835gnxip369_ref_0.jpg \
--seed 5876 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1

python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-20T16-56-07_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1hhj6mxfp90835n3wofrap.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1hhj6mxfp90835n3wofrap.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju2zp89k9q1g0855k1x0f1xa_ref_0.jpg \
--seed 5065 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1