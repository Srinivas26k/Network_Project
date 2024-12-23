import networkx as nx
import pandas as pd
import numpy as np

def preprocess_data(df):
    # Remove duplicates and reset index
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Ensure all names are strings and strip whitespace
    df['Name'] = df['Name'].astype(str).str.strip()
    df['Choose your pick'] = df['Choose your pick'].astype(str).str.strip()
    
    return df

def create_network(df):
    G = nx.Graph()
    
    # Add nodes
    for name in df['Name'].unique():
        G.add_node(name)
    
    # Add edges
    for _, row in df.iterrows():
        G.add_edge(row['Name'], row['Choose your pick'])
    
    return G

def create_embeddings(names):
    # Simple embedding function (not using external API)
    return np.random.rand(len(names), 768)

def query_faiss(index, query, names, k=5):
    # Simple query function (not using external API)
    query_vector = np.random.rand(768)
    _, indices = index.search(query_vector.reshape(1, -1), k)
    return [names[i] for i in indices[0]]

def generate_report(G, df, analysis_type):
    report_data = {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Average Degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "Number of Connected Components": nx.number_connected_components(G),
    }
    
    if analysis_type == "Community Detection":
        communities = nx.community.louvain_communities(G)
        report_data["Number of Communities"] = len(communities)
    
    return pd.DataFrame([report_data])

