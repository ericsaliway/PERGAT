# PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions
This repository provides the code for our research project "PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions".



## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) https://www.biosino.org/dbDEMC/index


## Scripts


## Setup

-) conda create -n gnn python=3.11 -y

-) conda activate gnn 

-) conda install pytorch::pytorch torchvision torchaudio -c pytorch

-) pip install pandas

-) pip install py2neo pandas matplotlib scikit-learn

-) pip install tqdm

-) conda install -c dglteam dgl

-) pip install seaborn

##
pip install -r requirements.txt


## Get start
## get embedding
python PERGAT_embedding/gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 105

PERGAT_embbedding % python gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 107

## prediction
python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

