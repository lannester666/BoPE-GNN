This repository contains the code implementation of the paper "Efficient Topology Encoding for Graph Neural
Network on Heterophilic Graphs". To reproduce the results in the paper, you need to follow these steps:

1. Download the datasets from https://github.com/CUAI/Non-Homophily-Large-Scale.git and https://github.com/RecklessRonan/GloGNN.git
2. Use 

```python
mkdir save_embd
python main.py --dataset snap-patents --directed --method gcnt --adj_order 3 --tebd_type ours --tebd_dim 512 --treduc_save 
```
to precompute the needed feature.

3.Use 
```python
python main.py --dataset snap-patents --directed  --device cuda:1 --adj_order 1  --tebd_dim 512 --hidden_channels 32  --method gat_pc  --w1 4 --tdropout 0.5 --dropout 0.5 --input_dropout 0.5  --tebd_type ours --runs 1 --epochs 300 --display_step 25 --gat_heads 4
```
to reproduce the reported result.
# BoPEGNN
