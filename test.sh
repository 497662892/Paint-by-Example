python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-28T09-56-41_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1f79yhsb5w0993txub59ol.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1f79yhsb5w0993txub59ol.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju1dhfok4mhe0878jlgrag0h_ref_0.jpg \
--seed 321 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1

python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-28T09-56-41_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1f79yhsb5w0993txub59ol.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1f79yhsb5w0993txub59ol.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju2trbpkv0c00988hxla5dzz_ref_0.jpg\
--seed 586 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1

python scripts/inference.py \
--outdir infers/Kvasir-SEG \
--config configs/polyp.yaml \
--ckpt /home/majiajian/code/diffusion/Paint-by-Example/logs/Paint-by-Example/2023-06-28T09-56-41_polyp/checkpoints/last.ckpt \
--image_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/images/cju1f79yhsb5w0993txub59ol.jpg \
--mask_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/masks/cju1f79yhsb5w0993txub59ol.jpg \
--reference_path /home/majiajian/dataset/polyp/Kvasir-SEG/val/reference/cju1b75x63ddl0799sdp0i2j3_ref_0.jpg \
--seed 5065 \
--scale 5 \
--ddim_steps 200 \
--ddim_eta 1