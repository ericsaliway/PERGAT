import os
import pickle
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from src.dataset import Dataset
from src.network import Network
from src.model import GATModel
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from dgl.nn import GraphConv

import dgl

class GCNModel(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)
        self.conv_0 = GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding

def create_network(path_miRNA_csv, kge):
    graph = Network(path_miRNA_csv, kge)
    return graph

def save_to_disk(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.kge + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))

def _save_miRNA_to_csv(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    miRNA_data = {'miRNA': [node['miRNA'] for node in graph.graph_nx.nodes.values()]}
    df = pd.DataFrame(miRNA_data)
    csv_path = os.path.join(save_dir, 'miRNA_nodes.csv')
    df.to_csv(csv_path, index=False)

def create_graphs(save=True, data_dir='data/emb'):
    graph_train = create_network('data/mirna_p_value_results_dbDEMC_train.csv', 'emb_train')
    graph_test = create_network('data/mirna_p_value_results_dbDEMC_test.csv', 'emb_test')

    print('creating graph=======================\n')
    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embeddings(load_model=True, save=True, data_dir='data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Dataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified
    feat_drop = hyperparams['feat_drop']
    attn_drop = hyperparams['attn_drop']

    net = GATModel(
            in_feats=in_feats,  
            out_feats=out_feats, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            feat_drop=feat_drop, 
            attn_drop=attn_drop, 
            do_train=True
        ).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        from src.train import train
        model_path = train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

def _create_embeddings_gcn(load_model=True, save=True, data_dir='data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Dataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)

    dim_latent = hyperparams['out_feats']
    # dim_latent = hyperparams['dim_latent']
    num_layers = hyperparams['num_layers']
    
    net = GCNModel(dim_latent=dim_latent, num_layers=num_layers).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        from src.train import train
        model_path = train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in range(len(data)):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

