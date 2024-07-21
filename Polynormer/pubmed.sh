GPU=2

## homophilic datasets
python main.py --dataset PubMed --hidden_channels 128 --local_epochs 100 --global_epochs 1000 --lr 0.001 --runs 1 --local_layers 7 --global_layers 2 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $GPU --save_model

