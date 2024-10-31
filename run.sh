#!/usr/bin bash
for s in 62
do
      python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --gpu_id 0\
        --beta 0.6\
        --lr 5e-5\
        --lr_pre 5e-5\
        --train_batch_size 512\
        --name clinc30_k1_bz512_lr5e-5_lrp5e-5_seed${s}_aug2_augp\
        --freeze_bert_parameters_em\
        --freeze_bert_parameters_pretrain\
        --k 1\
        --save_model \
        --augment_data \
        --pretrain \
        --num_pretrain_epochs 100\
        --num_train_epochs 100
done


