# PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions
This repository provides the code for our research project "PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions".



## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.

-) https://www.biosino.org/dbDEMC/index


## Setup

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.



## Get start
## get embedding
python PERGAT_embedding/gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 105

## prediction
python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

