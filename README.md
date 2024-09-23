# A Biological Knowledge Graph for Representational Learning 
This repository provides the code for our research project "A Biological Knowledge Graph for Representational Learning".



edge_prediction_project/
│
├── data/
│   └── neo4j_graph.json
│
├── results/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── utils.py
│   └── train.py
│
└── main.py


## python link_prediction_gat/main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) https://www.biosino.org/dbDEMC/index


## Scripts


## Setup
-)conda create -n kg python=3.10 -y

-)conda activate kg

-)pip install -r requirements.txt


## Get start

python link_prediction_gat/main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

#### residual=true
python link_prediction_gat/main.py --out-feats 128 --num-heads 2 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 503

Input Features  ----> Linear Transformation ----> Attention Mechanism ----> Message Passing ----> Output Features
      \                                                                                                 /
       \                                                                                               /
        ------------------------------------ Residual Connection --------------------------------------


####
    +-----------------+         +-----------------+         +-----------------+
   |                 |         |                 |         |                 |
   | Input Features  | ------> | Linear          | ------> | Attention       |
   |                 |         | Transformation  |         | Mechanism       |
   |                 |         |                 |         |                 |
   +-----------------+         +-----------------+         +-----------------+
                                                                 |
                                                                 V
   +-----------------+         +-----------------+         +-----------------+
   |                 |         |                 |         |                 |
   | Dropout &       | ------> | Message Passing | ------> | Residual        |
   | Leaky ReLU      |         |                 |         | Connection      |
   |                 |         |                 |         |                 |
   +-----------------+         +-----------------+         +-----------------+
                                                                 |
                                                                 V
                                                           +-----------------+
                                                           |                 |
                                                           | Activation &    |
                                                           | Bias Addition   |
                                                           |                 |
                                                           +-----------------+
                                                                 |
                                                                 V
                                                           +-----------------+
                                                           |                 |
                                                           | Link Prediction |
                                                           | (e.g., Dot      |
                                                           | Product, etc.)  |
                                                           |                 |
                                                           +-----------------+



## PASS !!
Test AUC: 0.7640 ± 0.0105 | Test F1: 0.7371 ± 0.0080 | Test Precision: 0.6462 ± 0.0086 | Test Recall: 0.8578 ± 0.0114 | Test mAP: 0.7352 ± 0.0127
(kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python link_prediction_gat/main.py --out-feats 32 --num-layers 4 --num-heads 4 --hidden-size 16 --input-size 2 --lr 0.01 --epochs 100


                       +-------------------+
                       | Initialize Module |
                       |                   |
                       |   - in_feats      |
                       |   - out_feats     |
                       |   - num_heads     |
                       |   - feat_drop     |
                       |   - attn_drop     |
                       |   - negative_slope|
                       |   - residual      |
                       |   - activation    |
                       |   - allow_zero_in_degree|
                       |   - bias          |
                       +---------+---------+
                                 |
                                 v
+------------------+        +----+----+        +-----------------------+
| Dropout (feat)   | -----> | Linear  | -----> | Attention Mechanism   |
|                  |        |  (fc)   |        | - attn_l, attn_r      |
|                  |        +---------+        +-----------------------+
|                  |                          /          \
+------------------+                         /            \
                                            /              \
                    +----------------+     /                \     +-------------------+
                    | Input Features | -->/    +------------+--> | Update Graph Data  |
                    +----------------+   /     | LeakyReLU  |    +-------------------+
                                          \    +------------+      - attn_drop
                                           \                        - edge_softmax
                                            \                     +-------------------+
                                             \                    | Message Passing   |
                                              \                   | - fn.u_mul_e      |
                                               \                  | - fn.sum          |
                                                \                 +-------------------+
                                                 \
                       +-------------------+      \
                       | Residual Connection|      \
                       +-------------------+       \
                       | - res_fc          |        \
                       +-------------------+         \
                                                      \
                       +-------------------+           \
                       | Bias Addition     |            \
                       +-------------------+             \
                       | - bias            |              \
                       +-------------------+               \
                                                          +-------------------+
                                                          | Activation        |
                                                          +-------------------+
                                                          | - activation      |
                                                          +-------------------+
                                                                  |
                                                                  v
                                                         +-------------------+
                                                         | Output Features   |
                                                         +-------------------+




+---------------------------------------------------------------------------------------+
|                                Link Prediction Process                                |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|              +-------------------+       +-------------------+       +--------------+  |
|              |    Graph Data     |       |  Node Embeddings  |       | Link Scores  |  |
|              |                   |       |      (Vectors)    |       |    (Scores)  |  |
|              +-------------------+       +-------------------+       +--------------+  |
|                       |                           |                          |         |
|                       |        +------------------+---------------------------+         |
|                       |        |                          |                          |         |
|      +----------------+--------v------------------+ +-----v-------------+   +-------v----------+|
|      |           Graph Construction     |        | |   Embedding Model   |   |  Link Prediction  ||
|      |   (Builds pathway graphs)        +--------> (e.g., GAT, GCN, etc)|   |  (e.g., MLP)      ||
|      +--------------------------------+        | +---------------------+   +-------------------+
|                       |        ^                          |                          |         |
|                       |        |                          |                          |         |
|         +-------------+--------+--------------------------+--------------------------+---------+
|         |                                                                             |
|  +------+-------------+                +---------------------------------------------+|
|  |   Node Features   |                |                    Graph Features            ||
|  |                   |                |                    (Pathway-Level)           ||
|  +-------------------+                +---------------------------------------------+
|                       |                           |                          |         |
|         +-------------+---------------------------+--------------------------+---------+
|         |                                                                             |
|  +------+-------------+                +-----------------+                +--------------+
|  |  Node Embeddings  |                |  Clustering Alg  |                |  Clustered   |
|  |                   |                |   (e.g., K-means) |                |   Nodes      |
|  +-------------------+                +-----------------+                +--------------+
|                       |                           |                          |         |
+-----------------------+---------------------------+--------------------------+---------+
