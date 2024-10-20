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
from src.dataset import miRNADataset
from src.model import GATModel
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
from src.plot import (create_heatmap_with_miRNA,plot_cosine_similarity_matrix_for_clusters_with_values,
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
    '''neo4j_uri = "neo4j+s://bb7d3bb8.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "0vZCoYqO6E9YkZRSFsdKPwHcziXu1-b0h8O9edAzWjM"

    reactome_file_path = "gat/data/NCBI2Reactome.csv"
    output_file_path = "gat/data/NCBI_pathway_map.csv"
    gene_names_file_path = "gat/data/gene_names.csv"
    pathway_map = create_pathway_map(reactome_file_path, output_file_path)
    gene_id_to_name_mapping, gene_id_to_symbol_mapping = read_gene_names(gene_names_file_path)'''
    
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = miRNADataset(data_path)
    print('ds==================\n',ds)
    
    ## "Cluster labels and number of nodes must match"
    ## make sure using the same training set
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    
    ## convert to dgl graph
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    net = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
    best_model

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
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
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
        first_node_miRNA_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        miRNA_dic_initial= {}

        # Populate miRNA_dic with node miRNAs mapped to embeddings
        for node in nx_graph.nodes:
            miRNA_dic_initial[nx_graph.nodes[node]['miRNA']] = node_embeddings_initial[node_to_index_initial[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if cluster not in first_node_miRNA_in_cluster_initial:
                first_node_miRNA_in_cluster_initial[cluster] = nx_graph.nodes[node]['miRNA']
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        ##print('first_node_miRNA_in_cluster_initial-------------------------------\n', first_node_miRNA_in_cluster_initial)
        miRNA_list = list(first_node_miRNA_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        ## print('embedding_list_initial-------------------\n',embedding_list_initial)
        create_heatmap_with_miRNA(embedding_list_initial, miRNA_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, miRNA_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, miRNA_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, miRNA_list, save_path_pca_initial)
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
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
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
                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1 = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1)
                
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
            print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

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
    save_path_heatmap_= os.path.join(results_path, f'heatmap_miRNA_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix = os.path.join(results_path, f'matrix_miRNA_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    
    cluster_miRNA_dict = {}  # Dictionary to store clusters and corresponding miRNAs
    significant_miRNAs = []  # List to store significant miRNAs
    clusters_with_significant_miRNA = {}  # Dictionary to store clusters and corresponding significant miRNAs
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        ## assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_miRNA_in_cluster = {}
        first_node_embedding_in_cluster = {}

        miRNA_dic = {}

        # Populate miRNA_dic with node miRNAs mapped to embeddings
        for node in nx_graph.nodes:
            miRNA_dic[nx_graph.nodes[node]['miRNA']] = node_embeddings[node_to_index[node]]
            # Check if the node's significance is 'significant' and add its miRNA to the list
            if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
                significant_miRNAs.append(nx_graph.nodes[node]['miRNA'])
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if cluster not in first_node_miRNA_in_cluster:
                first_node_miRNA_in_cluster[cluster] = nx_graph.nodes[node]['miRNA']
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                
            # Populate cluster_miRNA_dict
            if cluster not in cluster_miRNA_dict:
                cluster_miRNA_dict[cluster] = []
            cluster_miRNA_dict[cluster].append(nx_graph.nodes[node]['miRNA'])

            # Populate clusters_with_significant_miRNA
            if cluster not in clusters_with_significant_miRNA:
                clusters_with_significant_miRNA[cluster] = []
            if nx_graph.nodes[node]['miRNA'] in significant_miRNAs:
                clusters_with_significant_miRNA[cluster].append(nx_graph.nodes[node]['miRNA'])
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'miRNA': nx_graph.nodes[node]['miRNA'],
                'significance': graph.ndata['significance'][node_to_index[node]].item(),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
        
        print(first_node_miRNA_in_cluster)
        miRNA_list = list(first_node_miRNA_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
        create_heatmap_with_miRNA(embedding_list, miRNA_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, miRNA_list, save_path_matrix)

        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, miRNA_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, miRNA_list, save_path_pca)
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

    '''graph_train, graph_test = utils.create_embedding_with_markers()  

    # Get miRNA_mapping from save_graph_to_neo4j
    miRNA_mapping = utils.get_miRNA_mapping(graph_train)'''
    
    # Save significant miRNAs to JSON
    clusters_info_path = os.path.join(results_path, 'clusters_info.json')
    with open(clusters_info_path, 'w') as f:
        json.dump(significant_miRNAs, f)

    # Save cluster_miRNA_dict to JSON
    cluster_json_path = os.path.join(results_path, 'clusters.json')
    cluster_miRNA_dict_str_keys = {str(k): v for k, v in cluster_miRNA_dict.items()}
    with open(cluster_json_path, 'w') as f:
        json.dump(cluster_miRNA_dict_str_keys, f)

    # Save clusters_with_significant_miRNA to JSON
    clusters_with_significant_miRNA_path = os.path.join(results_path, 'clusters_with_significant_miRNA.json')
    clusters_with_significant_miRNA_str_keys = {str(k): v for k, v in clusters_with_significant_miRNA.items()}
    with open(clusters_with_significant_miRNA_path, 'w') as f:
        json.dump(clusters_with_significant_miRNA_str_keys, f)

    # Save clusters_node_info to JSON
    clusters_node_info_path = os.path.join(results_path, 'clusters_node_info.json')
    clusters_node_info_str_keys = {str(k): v for k, v in clusters_node_info.items()}
    with open(clusters_node_info_path, 'w') as f:
        json.dump(clusters_node_info_str_keys, f)

    ## save_to_neo4j(graph_train, miRNA_dic, miRNA_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
    miRNA_embeddings = pd.DataFrame.from_dict(miRNA_dic, orient='index')
    miRNA_embeddings.to_csv('gat/data/miRNA_embeddings.csv', index_label='miRNA')


    return model_path


def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="Pathway-pathway similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path)
    ##plt.show()
    plt.close()
    
def create_heatmap_with_miRNA(embedding_list, miRNA_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def create_heatmap_with_miRNA_ori(embedding_list, miRNA_list, save_path):
    # Create DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Transpose DataFrame to rotate 90 degrees clockwise
    heatmap_data = heatmap_data.T

    # Generate distinct colors using Seaborn color palette
    num_colors = len(miRNA_list)
    palette = sns.color_palette('dark', num_colors)  # Using 'dark' palette for distinct colors
    color_dict = {miRNA: palette[i] for i, miRNA in enumerate(miRNA_list)}

    plt.figure(figsize=(10, 10))  # Square figure size
    ax = sns.heatmap(heatmap_data, cmap='tab20', cbar_kws={'label': 'Mean embedding value', 'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04})
    
    plt.xlabel('Human pathways')
    plt.ylabel('Dimension of the embeddings', labelpad=0) 
    
    # Customize the color bar to be small and on top
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=8)
    cbar.set_label('Mean embedding value', size=10)
    cbar.ax.set_position([0.2, 0.92, 0.6, 0.03])  # [left, bottom, width, height]

    # Add custom color patches above each column
    for i, miRNA in enumerate(miRNA_list):
        color = color_dict[miRNA]
        rect = mpatches.Rectangle((i, heatmap_data.shape[0]), 1, 2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)

    # Adjust the axis limits to make space for the patches
    ax.set_xlim(0, len(miRNA_list))
    ax.set_ylim(0, heatmap_data.shape[0] + 1.5)

    # Custom x-axis labels with shorter ticks
    ax.set_xticks(np.arange(len(miRNA_list)) + 0.5)
    ax.set_xticklabels(miRNA_list, rotation=45, fontsize=8, ha='right')
    ax.tick_params(axis='x', length=5)  # Shorten the x-axis ticks

    # Custom y-axis labels, only plot the first and last dimension numbers
    y_tick_labels = [heatmap_data.index[0], heatmap_data.index[-1]]
    y_ticks = [0.5, len(heatmap_data.index) - 0.5]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.tick_params(axis='y', length=0)  # Remove y-axis ticks

    plt.savefig(save_path)
    plt.close()

def create_heatmap_with_miRNA_(embedding_list, miRNA_list, save_path):
    # Create DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Transpose DataFrame to rotate 90 degrees clockwise
    heatmap_data = heatmap_data.T

    # Generate distinct colors using Seaborn color palette
    num_colors = len(miRNA_list)
    palette = sns.color_palette('dark', num_colors)  # Using 'dark' palette for distinct colors
    color_dict = {miRNA: palette[i] for i, miRNA in enumerate(miRNA_list)}

    plt.figure(figsize=(10, 10))  # Square figure size
    ax = sns.heatmap(heatmap_data, cmap='tab20', cbar_kws={'label': 'Mean embedding value', 'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04})
    
    plt.xlabel('Human pathways')
    plt.ylabel('Dimension of the embeddings', labelpad=0) 
    
    # Customize the color bar to be small and on top
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=8)
    cbar.set_label('Mean embedding value', size=10)
    cbar.ax.set_position([0.2, 0.92, 0.6, 0.03])  # [left, bottom, width, height]

    # Add custom color patches above each column
    for i, miRNA in enumerate(miRNA_list):
        color = color_dict[miRNA]
        rect = mpatches.Rectangle((i, heatmap_data.shape[0]), 1, 2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)

    # Adjust the axis limits to make space for the patches
    ax.set_xlim(0, len(miRNA_list))
    ax.set_ylim(0, heatmap_data.shape[0] + 1.5)

    # Custom x-axis labels with shorter ticks
    ax.set_xticks(np.arange(len(miRNA_list)) + 1.0)
    ax.set_xticklabels(miRNA_list, rotation=30, fontsize=8, ha='right')
    ax.tick_params(axis='x', length=5)  # Shorten the x-axis ticks

    # Custom y-axis labels, only plot the first and last dimension numbers
    y_tick_labels = [heatmap_data.index[0], heatmap_data.index[-1]]
    y_ticks = [0.5, len(heatmap_data.index) - 0.5]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.tick_params(axis='y', length=0)  # Remove y-axis ticks

    plt.savefig(save_path)
    plt.close()   

def calculate_cluster_labels(net, dataloader, device, num_clusters=20):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels


def visualize_embeddings_pca(embeddings, cluster_labels, miRNA_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_pca_ori(embeddings, cluster_labels, miRNA_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    
def visualize_embeddings_tsne(embeddings, cluster_labels, miRNA_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def export_to_cytoscape(node_embeddings, cluster_labels, miRNA_list, output_path):
    # Create a DataFrame for Cytoscape export
    data = {
        'Node': miRNA_list,
        'Cluster': cluster_labels,
        'Embedding': list(node_embeddings)
    }
    df = pd.DataFrame(data)
    
    # Expand the embedding column into separate columns
    embeddings_df = pd.DataFrame(node_embeddings, columns=[f'Embed_{i}' for i in range(node_embeddings.shape[1])])
    df = df.drop('Embedding', axis=1).join(embeddings_df)

    # Save to CSV for Cytoscape import
    df.to_csv(output_path, index=False)
    print(f"Data exported to {output_path} for Cytoscape visualization.")


def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()


def train_(hyperparams=None, data_path='gat/data/emb', plot=True):
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
    
    ds = miRNADataset(data_path)
    ##print('ds==================\n',ds)
    
    ## "Cluster labels and number of nodes must match"
    ## make sure using the same training set
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    
    ## convert to dgl graph
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    '''net = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))'''
    net = GATModel(out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = GATModel(out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
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
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
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
        first_node_miRNA_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        miRNA_dic_initial= {}

        # Populate miRNA_dic with node miRNAs mapped to embeddings
        for node in nx_graph.nodes:
            miRNA_dic_initial[nx_graph.nodes[node]['miRNA']] = node_embeddings_initial[node_to_index_initial[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if cluster not in first_node_miRNA_in_cluster_initial:
                first_node_miRNA_in_cluster_initial[cluster] = nx_graph.nodes[node]['miRNA']
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        ##print('first_node_miRNA_in_cluster_initial-------------------------------\n', first_node_miRNA_in_cluster_initial)
        miRNA_list = list(first_node_miRNA_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        ## print('embedding_list_initial-------------------\n',embedding_list_initial)
        create_heatmap_with_miRNA(embedding_list_initial, miRNA_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, miRNA_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, miRNA_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, miRNA_list, save_path_pca_initial)
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
    save_path_heatmap_= os.path.join(results_path, f'heatmap_miRNA_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix = os.path.join(results_path, f'matrix_miRNA_lr{learning_rate}_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    
    cluster_miRNA_dict = {}  # Dictionary to store clusters and corresponding miRNAs
    significant_miRNAs = []  # List to store significant miRNAs
    clusters_with_significant_miRNA = {}  # Dictionary to store clusters and corresponding significant miRNAs
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        ## assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_miRNA_in_cluster = {}
        first_node_embedding_in_cluster = {}

        miRNA_dic = {}

        # Populate miRNA_dic with node miRNAs mapped to embeddings
        for node in nx_graph.nodes:
            miRNA_dic[nx_graph.nodes[node]['miRNA']] = node_embeddings[node_to_index[node]]
            # Check if the node's significance is 'significant' and add its miRNA to the list
            if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
                significant_miRNAs.append(nx_graph.nodes[node]['miRNA'])
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if cluster not in first_node_miRNA_in_cluster:
                first_node_miRNA_in_cluster[cluster] = nx_graph.nodes[node]['miRNA']
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                
            # Populate cluster_miRNA_dict
            if cluster not in cluster_miRNA_dict:
                cluster_miRNA_dict[cluster] = []
            cluster_miRNA_dict[cluster].append(nx_graph.nodes[node]['miRNA'])

            # Populate clusters_with_significant_miRNA
            if cluster not in clusters_with_significant_miRNA:
                clusters_with_significant_miRNA[cluster] = []
            if nx_graph.nodes[node]['miRNA'] in significant_miRNAs:
                clusters_with_significant_miRNA[cluster].append(nx_graph.nodes[node]['miRNA'])
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'miRNA': nx_graph.nodes[node]['miRNA'],
                'significance': graph.ndata['significance'][node_to_index[node]].item(),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
        
        print(first_node_miRNA_in_cluster)
        miRNA_list = list(first_node_miRNA_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
        create_heatmap_with_miRNA(embedding_list, miRNA_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, miRNA_list, save_path_matrix)

        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, miRNA_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, miRNA_list, save_path_pca)
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

    # Get miRNA_mapping from save_graph_to_neo4j
    ##miRNA_mapping = get_miRNA_mapping(graph_train)
    
    # Save significant miRNAs to JSON
    clusters_info_path = os.path.join(results_path, 'clusters_info.json')
    with open(clusters_info_path, 'w') as f:
        json.dump(significant_miRNAs, f)

    # Save cluster_miRNA_dict to JSON
    cluster_json_path = os.path.join(results_path, 'clusters.json')
    cluster_miRNA_dict_str_keys = {str(k): v for k, v in cluster_miRNA_dict.items()}
    with open(cluster_json_path, 'w') as f:
        json.dump(cluster_miRNA_dict_str_keys, f)

    # Save clusters_with_significant_miRNA to JSON
    clusters_with_significant_miRNA_path = os.path.join(results_path, 'clusters_with_significant_miRNA.json')
    clusters_with_significant_miRNA_str_keys = {str(k): v for k, v in clusters_with_significant_miRNA.items()}
    with open(clusters_with_significant_miRNA_path, 'w') as f:
        json.dump(clusters_with_significant_miRNA_str_keys, f)

    # Save clusters_node_info to JSON
    clusters_node_info_path = os.path.join(results_path, 'clusters_node_info.json')
    clusters_node_info_str_keys = {str(k): v for k, v in clusters_node_info.items()}
    with open(clusters_node_info_path, 'w') as f:
        json.dump(clusters_node_info_str_keys, f)

    ## save_to_neo4j(graph_train, miRNA_dic, miRNA_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
    miRNA_embeddings = pd.DataFrame.from_dict(miRNA_dic, orient='index')
    miRNA_embeddings.to_csv('gat/data/miRNA_embeddings.csv', index_label='miRNA')


    return model_path
def _train(hyperparams=None, data_path='gat/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = miRNADataset(data_path)

    ds_train = [ds[1]]
    ds_valid = [ds[0]]

    
    ## convert to dgl graph
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    net = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
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
    
    ##all_miRNA_embeddings, miRNA_cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names = (net, dl_train, device, 4)
    all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, graph_name_initial_miRNA, all_embeddings_initial_disease, cluster_labels_initial_disease, graph_name_initial_disease = calculate_cluster_labels(best_model, dl_train, device)

    ##all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, graph_name_initial_miRNA = calculate_cluster_labels_miRNA(best_model, dl_train, device)
    ##print('cluster_labels_initial--------------------------\n',cluster_labels_initial)
    all_embeddings_initial_miRNA = all_embeddings_initial_miRNA.reshape(all_embeddings_initial_miRNA.shape[0], -1)  # Flatten 
    save_path_heatmap_initial_miRNA = os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial_miRNA = os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial_miRNA = os.path.join(results_path, f'pca_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial_miRNA = os.path.join(results_path, f't-SNE_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')

    ##all_embeddings_initial_disease, cluster_labels_initial_disease, graph_name_initial_disease = calculate_cluster_labels_disease(best_model, dl_train, device)
    ##print('cluster_labels_initial--------------------------\n',cluster_labels_initial)
    all_embeddings_initial_disease = all_embeddings_initial_disease.reshape(all_embeddings_initial_disease.shape[0], -1)  # Flatten 
    save_path_heatmap_initial_disease = os.path.join(results_path, f'heatmap_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial_disease = os.path.join(results_path, f'matrix_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial_disease = os.path.join(results_path, f'pca_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial_disease = os.path.join(results_path, f't-SNE_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
                
    for data in dl_train:
        graph, _ = data
        # Get all node embeddings from the graph
        node_embeddings_initial = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        # Extract the 'node_type' attribute from the graph
        node_types = graph.ndata['node_type'].cpu().numpy()
        # Create a mask for miRNA nodes (assuming 'miRNA' is represented as a string in node_type)
        miRNA_mask = (node_types == 0)

        # Filter the embeddings using the miRNA mask
        miRNA_node_embeddings = node_embeddings_initial[miRNA_mask]
        disease_mask = (node_types == 1)

        # Filter the embeddings using the disease mask
        disease_node_embeddings = node_embeddings_initial[disease_mask]
                
        ##graph_path = os.path.join(data_path, f'raw/{graph_name_initial_miRNA}.pkl')
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        # Filter and get all miRNA nodes
        miRNA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'miRNA']

        ##assert len(cluster_labels_initial_miRNA) == len(miRNA_nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial_miRNA = {node: idx for idx, node in enumerate(miRNA_nodes)}

        disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'disease']
    
        ##assert len(cluster_labels_initial_disease) == len(disease_nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial_disease = {node: idx for idx, node in enumerate(disease_nodes)}
                
        miRNA_dic_initial, first_node_miRNA_in_cluster_initial, first_node_embedding_in_cluster_initial_miRNA = populate_miRNA_dic(nx_graph, miRNA_node_embeddings, node_to_index_initial_miRNA, cluster_labels_initial_miRNA)
        disease_dic_initial, first_node_disease_in_cluster_initial, first_node_embedding_in_cluster_initial_disease = populate_disease_dic(nx_graph, disease_node_embeddings, node_to_index_initial_disease, cluster_labels_initial_disease)

        ##print('first_node_miRNA_in_cluster_initial-------------------------------\n', first_node_miRNA_in_cluster_initial)
        miRNA_list = list(first_node_miRNA_in_cluster_initial.values())
        miRNA_embedding_list_initial = list(first_node_embedding_in_cluster_initial_miRNA.values())
        
        disease_list = list(first_node_disease_in_cluster_initial.values())
        disease_embedding_list_initial = list(first_node_embedding_in_cluster_initial_disease.values())

        disease_plot_cosine_similarity_matrix_for_clusters_with_values(disease_embedding_list_initial, disease_list, save_path_matrix_initial_disease)
        miRNA_plot_cosine_similarity_matrix_for_clusters_with_values(miRNA_embedding_list_initial, miRNA_list, save_path_matrix_initial_miRNA)
        
        create_heatmap(disease_embedding_list_initial, disease_list, save_path_heatmap_initial_disease)
        create_heatmap(miRNA_embedding_list_initial, miRNA_list, save_path_heatmap_initial_miRNA)

        
        break

    visualize_embeddings_tsne(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, miRNA_list, save_path_t_SNE_initial_miRNA)
    #visualize_embeddings_pca(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, miRNA_list, save_path_pca_initial_miRNA)
    silhouette_avg_ = silhouette_score(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'miRNA_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)


    visualize_embeddings_tsne(all_embeddings_initial_disease, cluster_labels_initial_disease, disease_list, save_path_t_SNE_initial_disease)
    #visualize_embeddings_pca(all_embeddings_initial_disease, cluster_labels_initial_disease, disease_list, save_path_pca_initial_disease)
    silhouette_avg_ = silhouette_score(all_embeddings_initial_disease, cluster_labels_initial_disease)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial_disease, cluster_labels_initial_disease)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'disease_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)    
            
    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []    
    accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []
    
    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = []
            accuracy_per_graph = []
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
                preds = (logits.sigmoid() > 0.5).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                accuracy = metrics.accuracy_score(labels, preds)
                f1_per_graph.append(f1)
                accuracy_per_graph.append(accuracy)

            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            running_accuracy = np.array(accuracy_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)
            accuracy_per_epoch_train.append(running_accuracy)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph_val = []
                accuracy_per_graph_val = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    preds = (logits.sigmoid() > 0.5).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    accuracy = metrics.accuracy_score(labels, preds)
                    f1_per_graph_val.append(f1)
                    accuracy_per_graph_val.append(accuracy)

                running_loss = np.array(loss_per_graph).mean()
                running_f1_val = np.array(f1_per_graph_val).mean()
                running_accuracy_val = np.array(accuracy_per_graph_val).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1_val)
                accuracy_per_epoch_valid.append(running_accuracy_val)

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
            print(f"Epoch {epoch + 1} - F1 Train: {running_f1}, F1 Valid: {running_f1_val}, Accuracy Train: {running_accuracy}, Accuracy Valid: {running_accuracy_val}")

    ##all_embeddings, cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_miRNA, cluster_labels_miRNA, graph_name_initial_miRNA, all_embeddings_disease, cluster_labels_disease, graph_name_disease = calculate_cluster_labels(best_model, dl_train, device)
    
    all_embeddings_miRNA = all_embeddings_miRNA.reshape(all_embeddings_miRNA.shape[0], -1)  # Flatten 
    save_path_heatmap_miRNA = os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix_miRNA = os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_pca_miRNA = os.path.join(results_path, f'pca_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE_miRNA = os.path.join(results_path, f't-SNE_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

    ##all_embeddings_disease, cluster_labels_disease, graph_name_disease = calculate_cluster_labels_disease(best_model, dl_train, device)
    ##print('cluster_labels--------------------------\n',cluster_labels)
    all_embeddings_disease = all_embeddings_disease.reshape(all_embeddings_disease.shape[0], -1)  # Flatten 
    save_path_heatmap_disease = os.path.join(results_path, f'heatmap_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix_disease = os.path.join(results_path, f'matrix_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_pca_disease = os.path.join(results_path, f'pca_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE_disease = os.path.join(results_path, f't-SNE_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
                
                    
    all_embeddings_miRNA = all_embeddings_miRNA.reshape(all_embeddings_miRNA.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels_miRNA)

    cos_sim = np.dot(all_embeddings_miRNA, all_embeddings_miRNA.T)
    norms = np.linalg.norm(all_embeddings_miRNA, axis=1)
    cos_sim /= np.outer(norms, norms)
    
    if plot:
        loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        accuracy_path = os.path.join(results_path, f'accuracy_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)
        draw_accuracy_plot(accuracy_per_epoch_train, accuracy_per_epoch_valid, accuracy_path)
        
    torch.save(best_model.state_dict(), model_path)

    for data in dl_train:
        graph, _ = data
        # Get all node embeddings from the graph
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        # Extract the 'node_type' attribute from the graph
        node_types = graph.ndata['node_type'].cpu().numpy()
        
        print('node_types=====================\n',node_types)

        # Create a mask for miRNA nodes (assuming 'miRNA' is represented as a string in node_type)
        miRNA_mask = (node_types == 0)

        # Filter the embeddings using the miRNA mask
        miRNA_node_embeddings = node_embeddings[miRNA_mask]

        print(f"Shape of miRNA node embeddings: {miRNA_node_embeddings.shape}")

        disease_mask = (node_types == 1)

        # Filter the embeddings using the disease mask
        disease_node_embeddings = node_embeddings[disease_mask]

        print(f"Shape of disease node embeddings: {disease_node_embeddings.shape}")
                
        ##graph_path = os.path.join(data_path, f'raw/{graph_name_miRNA}.pkl')
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        # Filter and get all miRNA nodes
        miRNA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'miRNA']
    
        print('cluster_labels_miRNA=====================\n',cluster_labels_miRNA)
        print('len(miRNA_nodes)=====================\n',len(miRNA_nodes))

        ##assert len(cluster_labels_miRNA) == len(miRNA_nodes), "Cluster labels and number of nodes must match"
        node_to_index_miRNA = {node: idx for idx, node in enumerate(miRNA_nodes)}

        disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'disease']

        print('cluster_labels_disease=====================\n',cluster_labels_disease)
        print('len(disease_nodes)=====================\n',len(disease_nodes))
               
        ##assert len(cluster_labels_disease) == len(disease_nodes), "Cluster labels and number of nodes must match"
        node_to_index_disease = {node: idx for idx, node in enumerate(disease_nodes)}
                
        miRNA_dic, first_node_miRNA_in_cluster, first_node_embedding_in_cluster_miRNA = populate_miRNA_dic(nx_graph, miRNA_node_embeddings, node_to_index_miRNA, cluster_labels_miRNA)
        disease_dic, first_node_disease_in_cluster, first_node_embedding_in_cluster_disease = populate_disease_dic(nx_graph, disease_node_embeddings, node_to_index_disease, cluster_labels_disease)

        print('first_node_miRNA_in_cluster=====================\n',first_node_miRNA_in_cluster)
        print('first_node_disease_in_cluster=====================\n',first_node_disease_in_cluster)
                
        ##print('first_node_miRNA_in_cluster-------------------------------\n', first_node_miRNA_in_cluster)
        miRNA_list = list(first_node_miRNA_in_cluster.values())
        miRNA_embedding_list = list(first_node_embedding_in_cluster_miRNA.values())
        
        ##print('miRNA_embedding_list=====================\n',miRNA_embedding_list)
        print('miRNA_list=====================\n',miRNA_list)

        disease_list = list(first_node_disease_in_cluster.values())
        disease_embedding_list = list(first_node_embedding_in_cluster_disease.values())

        ##print('disease_embedding_list=====================\n',disease_embedding_list)
        print('disease_list=====================\n',disease_list)

        
        

        disease_plot_cosine_similarity_matrix_for_clusters_with_values(disease_embedding_list, disease_list, save_path_matrix_disease)
        miRNA_plot_cosine_similarity_matrix_for_clusters_with_values(miRNA_embedding_list, miRNA_list, save_path_matrix_miRNA)
        
        create_heatmap(disease_embedding_list, disease_list, save_path_heatmap_disease)
        create_heatmap(miRNA_embedding_list, miRNA_list, save_path_heatmap_miRNA)

        
        break

    visualize_embeddings_tsne(all_embeddings_miRNA, cluster_labels_miRNA, miRNA_list, save_path_t_SNE_miRNA)
    #visualize_embeddings_pca(all_embeddings_miRNA, cluster_labels_miRNA, miRNA_list, save_path_pca_miRNA)
    silhouette_avg_miRNA = silhouette_score(all_embeddings_miRNA, cluster_labels_miRNA)
    davies_bouldin_miRNA = davies_bouldin_score(all_embeddings_miRNA, cluster_labels_miRNA)
    summary_  = f"Silhouette Score: {silhouette_avg_miRNA}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_miRNA}\n"

    save_file_= os.path.join(results_path, f'miRNA_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)


    visualize_embeddings_tsne(all_embeddings_disease, cluster_labels_disease, disease_list, save_path_t_SNE_disease)
    #visualize_embeddings_pca(all_embeddings_disease, cluster_labels_disease, disease_list, save_path_pca_disease)
    silhouette_avg_disease = silhouette_score(all_embeddings_disease, cluster_labels_disease)
    davies_bouldin_disease = davies_bouldin_score(all_embeddings_disease, cluster_labels_disease)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'disease_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)    

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg_disease}")
    print(f"Davies-Bouldin Index: {davies_bouldin_disease}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg_disease}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin_disease}\n"

    save_file = os.path.join(results_path, f'head{num_heads}_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)


    ## save_to_neo4j(graph_train, miRNA_dic, miRNA_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
    
    miRNA_embeddings_initial = pd.DataFrame.from_dict(miRNA_dic_initial, orient='index')
    miRNA_embeddings_initial.to_csv('gat/data/miRNA_embeddings_initial.csv', index_label='miRNA')

    miRNA_embeddings = pd.DataFrame.from_dict(miRNA_dic, orient='index')
    miRNA_embeddings.to_csv('gat/data/pretrain_miRNA_embeddings.csv', index_label='miRNA')
    
    disease_embeddings_initial = pd.DataFrame.from_dict(disease_dic_initial, orient='index')
    disease_embeddings_initial.to_csv('gat/data/disease_embeddings_initial.csv', index_label='disease')

    disease_embeddings = pd.DataFrame.from_dict(disease_dic, orient='index')
    disease_embeddings.to_csv('gat/data/pretrain_disease_embeddings.csv', index_label='disease')
    
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
