import math
import json
import urllib.request
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx
import pandas as pd
from py2neo import Graph, Node, Relationship
from networkx.algorithms.traversal.depth_first_search import dfs_tree


class Network:
    def __init__(self, csv_file_path, kge=None):

        if kge is not None:
            self.kge = kge
        else:
            time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
            kge = time_now
        
        self.data = self.load_csv(csv_file_path)
        
        ##self.weights = self.set_weights()

        self.graph_nx = self.to_networkx()

    def load_csv(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        return df
    
    def set_node_attributes(self):
        diseases = {}
        miRNAs = {}
        weights = {}
        significances = {}

        for _, row in self.data.iterrows():
            disease1 = row['disease1']
            disease2 = row['disease2']
            miRNA = row['shared_miRNAs']
            p_value = row['adjusted_p-value']
            ##p_value = row['fdr_corrected_p-value']
            significance = row['significance']

            diseases[disease1] = disease1
            diseases[disease2] = disease2

            if disease1 not in miRNAs:
                miRNAs[disease1] = miRNA
            if disease2 not in miRNAs:
                miRNAs[disease2] = miRNA

            if disease1 not in weights:
                weights[disease1] = p_value
            if disease2 not in weights:
                weights[disease2] = p_value

            if disease1 not in significances:
                significances[disease1] = significance
            if disease2 not in significances:
                significances[disease2] = significance

        return diseases, miRNAs, weights, significances

    def to_networkx(self):
        graph_nx = nx.Graph()
        ##graph_nx = nx.DiGraph()

        # Add nodes and edges with attributes
        for _, row in self.data.iterrows():
            disease1 = row['disease1']
            disease2 = row['disease2']
            miRNA = row['shared_miRNAs']
            p_value = row['adjusted_p-value']
            ##p_value = row['fdr_corrected_p-value']
            significance = row['significance']

            ##if disease > 0:
            graph_nx.add_node(disease1, miRNA=miRNA, p_value=p_value, significance=significance)
            graph_nx.add_node(disease2, miRNA=miRNA, p_value=p_value, significance=significance)
            graph_nx.add_edge(disease1, disease2, p_value=p_value, significance=significance)

        diseases, miRNAs, weights, significances = self.set_node_attributes()

        nx.set_node_attributes(graph_nx, diseases, 'disease')
        nx.set_node_attributes(graph_nx, miRNAs, 'miRNA')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        nx.set_node_attributes(graph_nx, significances, 'significance')

        return graph_nx

