from datetime import datetime
import networkx as nx
import pandas as pd


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
        miRNAs = {}
        diseases = {}
        weights = {}
        significances = {}

        for _, row in self.data.iterrows():
            miRNA1 = row['miRNA1']
            miRNA2 = row['miRNA2']
            disease = row['shared_diseases']
            p_value = row['adjusted_p-value']
            significance = row['significance']

            miRNAs[miRNA1] = miRNA1
            miRNAs[miRNA2] = miRNA2

            if miRNA1 not in diseases:
                diseases[miRNA1] = disease
            if miRNA2 not in diseases:
                diseases[miRNA2] = disease

            if miRNA1 not in weights:
                weights[miRNA1] = p_value
            if miRNA2 not in weights:
                weights[miRNA2] = p_value

            if miRNA1 not in significances:
                significances[miRNA1] = significance
            if miRNA2 not in significances:
                significances[miRNA2] = significance

        return miRNAs, diseases, weights, significances

    def to_networkx(self):
        graph_nx = nx.Graph()
        ##graph_nx = nx.DiGraph()

        # Add nodes and edges with attributes
        for _, row in self.data.iterrows():
            miRNA1 = row['miRNA1']
            miRNA2 = row['miRNA2']
            disease = row['shared_diseases']
            p_value = row['adjusted_p-value']
            significance = row['significance']

            ##if disease > 0:
            graph_nx.add_node(miRNA1, disease=disease, p_value=p_value, significance=significance)
            graph_nx.add_node(miRNA2, disease=disease, p_value=p_value, significance=significance)
            graph_nx.add_edge(miRNA1, miRNA2, p_value=p_value, significance=significance)

        miRNAs, diseases, weights, significances = self.set_node_attributes()

        nx.set_node_attributes(graph_nx, miRNAs, 'miRNA')
        nx.set_node_attributes(graph_nx, diseases, 'disease')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        nx.set_node_attributes(graph_nx, significances, 'significance')

        return graph_nx

