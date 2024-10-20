import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dataset
import model, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import torch
import torch.nn as nn
from src.utils import create_graphs
from src.plot import (create_heatmap_with_disease,plot_cosine_similarity_matrix_for_clusters_with_values,
                    visualize_embeddings_tsne,visualize_embeddings_pca,calculate_cluster_labels,draw_loss_plot,
                    draw_max_f1_plot,draw_f1_plot)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train(hyperparams=None, data_path='gat/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    neo4j_uri = "neo4j+s://bb7d3bb8.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "0vZCoYqO6E9YkZRSFsdKPwHcziXu1-b0h8O9edAzWjM"

    '''reactome_file_path = "gat/data/NCBI2Reactome.csv"
    output_file_path = "gat/data/NCBI_pathway_map.csv"
    gene_names_file_path = "gat/data/gene_names.csv"
    pathway_map = create_pathway_map(reactome_file_path, output_file_path)
    gene_id_to_name_mapping, gene_id_to_symbol_mapping = read_gene_names(gene_names_file_path)
    '''
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = dataset.PathwayDataset(data_path)
    ##print('ds==================\n',ds)
    
    ## "Cluster labels and number of nodes must match"
    ## make sure using the same training set
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    
    ## convert to dgl graph
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    ##criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = 'gat/results/node_embeddings/'
    os.makedirs(results_path, exist_ok=True)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    ##print('cluster_labels_initial--------------------------\n',cluster_labels_initial)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'matrix_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        print('len(cluster_labels_initial)=====================\n',len(cluster_labels_initial))
        print('len(nx_graph.nodes)=====================\n',len(nx_graph.nodes))

        assert len(cluster_labels_initial) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_disease_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        disease_dic_initial= {}

        # Populate disease_dic with node diseases mapped to embeddings
        for node in nx_graph.nodes:
            disease_dic_initial[nx_graph.nodes[node]['disease']] = node_embeddings_initial[node_to_index_initial[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if cluster not in first_node_disease_in_cluster_initial:
                first_node_disease_in_cluster_initial[cluster] = nx_graph.nodes[node]['disease']
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        ##print('first_node_disease_in_cluster_initial-------------------------------\n', first_node_disease_in_cluster_initial)
        disease_list = list(first_node_disease_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        ## print('embedding_list_initial-------------------\n',embedding_list_initial)
        create_heatmap_with_disease(embedding_list_initial, disease_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, disease_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, disease_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, disease_list, save_path_pca_initial)
    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)
      
    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                preds = (logits.sigmoid() > 0.5).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)

            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph_val = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    ##logits = net(graph, graph.ndata['feat'])
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    preds = (logits.sigmoid() > 0.5).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    f1_per_graph_val.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1_val = np.array(f1_per_graph_val).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1_val)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - F1 Train: {running_f1}, F1 Valid: {running_f1_val}")

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels)

    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(results_path, f'max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(results_path, f'matrix_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
 
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)

    save_path_pca = os.path.join(results_path, f'pca_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE = os.path.join(results_path, f't-SNE_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap_= os.path.join(results_path, f'heatmap_disease_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix = os.path.join(results_path, f'matrix_disease_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    
    cluster_disease_dict = {}  # Dictionary to store clusters and corresponding diseases
    significant_diseases = []  # List to store significant diseases
    clusters_with_significant_disease = {}  # Dictionary to store clusters and corresponding significant diseases
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        ## assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_disease_in_cluster = {}
        first_node_embedding_in_cluster = {}

        disease_dic = {}

        # Populate disease_dic with node diseases mapped to embeddings
        for node in nx_graph.nodes:
            disease_dic[nx_graph.nodes[node]['disease']] = node_embeddings[node_to_index[node]]
            # Check if the node's significance is 'significant' and add its disease to the list
            if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
                significant_diseases.append(nx_graph.nodes[node]['disease'])
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if cluster not in first_node_disease_in_cluster:
                first_node_disease_in_cluster[cluster] = nx_graph.nodes[node]['disease']
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                
            # Populate cluster_disease_dict
            if cluster not in cluster_disease_dict:
                cluster_disease_dict[cluster] = []
            cluster_disease_dict[cluster].append(nx_graph.nodes[node]['disease'])

            # Populate clusters_with_significant_disease
            if cluster not in clusters_with_significant_disease:
                clusters_with_significant_disease[cluster] = []
            if nx_graph.nodes[node]['disease'] in significant_diseases:
                clusters_with_significant_disease[cluster].append(nx_graph.nodes[node]['disease'])
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'disease': nx_graph.nodes[node]['disease'],
                'significance': graph.ndata['significance'][node_to_index[node]].item(),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
        
        print(first_node_disease_in_cluster)
        disease_list = list(first_node_disease_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=disease_list)
        create_heatmap_with_disease(embedding_list, disease_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, disease_list, save_path_matrix)

        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, disease_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, disease_list, save_path_pca)
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"

    save_file = os.path.join(results_path, f'head{num_heads}_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)

    graph_train, graph_test = create_graphs()  

    '''# Get disease_mapping from save_graph_to_neo4j
    disease_mapping = utils.get_disease_mapping(graph_train)
    
    # Save significant diseases to JSON
    clusters_info_path = os.path.join(results_path, 'clusters_info.json')
    with open(clusters_info_path, 'w') as f:
        json.dump(significant_diseases, f)

    # Save cluster_disease_dict to JSON
    cluster_json_path = os.path.join(results_path, 'clusters.json')
    cluster_disease_dict_str_keys = {str(k): v for k, v in cluster_disease_dict.items()}
    with open(cluster_json_path, 'w') as f:
        json.dump(cluster_disease_dict_str_keys, f)

    # Save clusters_with_significant_disease to JSON
    clusters_with_significant_disease_path = os.path.join(results_path, 'clusters_with_significant_disease.json')
    clusters_with_significant_disease_str_keys = {str(k): v for k, v in clusters_with_significant_disease.items()}
    with open(clusters_with_significant_disease_path, 'w') as f:
        json.dump(clusters_with_significant_disease_str_keys, f)

    # Save clusters_node_info to JSON
    clusters_node_info_path = os.path.join(results_path, 'clusters_node_info.json')
    clusters_node_info_str_keys = {str(k): v for k, v in clusters_node_info.items()}
    with open(clusters_node_info_path, 'w') as f:
        json.dump(clusters_node_info_str_keys, f)'''

    ## save_to_neo4j(graph_train, disease_dic, disease_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
    disease_embeddings = pd.DataFrame.from_dict(disease_dic, orient='index')
    disease_embeddings.to_csv('gat/data/disease_embeddings.csv', index_label='disease')


    return model_path


if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
