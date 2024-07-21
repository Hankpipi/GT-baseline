CUDA_VISIBLE_DEVICES=2 python main-batch.py --method hsgt --dataset arxiv-year --metric acc --lr 0.001 --hidden_channels 128 \
    --num_horizontal_layer 2  --num_readout_layer 1 --trans_num_heads 8 --seed 123 --runs 1 --batch_size 4 --epochs 1000 --eval_step 9 \
    --shared_params True --device 0