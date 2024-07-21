CUDA_VISIBLE_DEVICES=2 python main-batch.py --method hsgt --dataset pubmed --metric acc --lr 0.001 --hidden_channels 128 \
    --num_horizontal_layer 3  --num_readout_layer 1 --trans_num_heads 8 --seed 123 --runs 1 --batch_size 32 --epochs 1000 --eval_step 9 \
    --shared_params False  --dropout 0.1 --device 0

CUDA_VISIBLE_DEVICES=2 python main-batch.py --method hsgt --dataset arxiv-year --metric acc --lr 0.001 --hidden_channels 128 \
    --num_horizontal_layer 2  --num_readout_layer 1 --trans_num_heads 8 --seed 123 --runs 1 --batch_size 4 --epochs 1000 --eval_step 9 \
    --shared_params True --device 0

CUDA_VISIBLE_DEVICES=1 python main-batch.py --method hsgt --dataset ogbn-products --lr 0.0001 --metric acc \
    --num_horizontal_layer 3  --num_readout_layer 1 --hidden_channels 128 --trans_num_heads 8 \
    --epochs 200 --seed 123 --batch_size 4 --eval_step 9 --device 0 --runs 1