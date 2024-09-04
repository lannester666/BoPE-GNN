This repository contains the code implementation of the paper "[Improved topology features for node classification on heterophilic graphs](https://rdcu.be/dSQVR)". To reproduce the results in the paper, you need to follow these steps:

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

If you find our work useful, please cite our paper by 
```
@InProceedings{10.1007/978-3-031-70368-3_7,
author="Lai, Yurui
and Zhang, Taiyan
and Fan, Rui",
editor="Bifet, Albert
and Davis, Jesse
and Krilavi{\v{c}}ius, Tomas
and Kull, Meelis
and Ntoutsi, Eirini
and {\v{Z}}liobait{\.{e}}, Indr{\.{e}}",
title="Improved Topology Features for Node Classification on Heterophilic Graphs",
booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="105--123",
isbn="978-3-031-70368-3"
}
```
