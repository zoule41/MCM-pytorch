CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain_2.py \
    --num_workers 8 \
    --accum_iter 2 \
    --batch_size 64 \
    --model mcm \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 250 \
    --warmup_epochs 40 \
    --lr 1.5e-4 --weight_decay 0.05 \
    --resume '/root/autodl-tmp/mae_pretrain_vit_base.pth' \
    --data_path '/root/autodl-tmp/data/mimic' \
    --output_dir '/root/autodl-tmp/project/MCM/output' \
