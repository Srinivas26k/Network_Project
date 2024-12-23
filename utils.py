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
    # Simple embedding function using character count
    return np.array([[len(name)] for name in names])

def query_faiss(index, query, names, k=5):
    query_vector = np.array([[len(query)]])  # Use the same embedding method as create_embeddings
    _, indices = index.search(query_vector, k)
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

def calculate_pagerank(G):
    return nx.pagerank(G)

def analyze_connection_quality(G, node):
    neighbors = list(G.neighbors(node))
    total_possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
    actual_connections = sum(1 for n1 in neighbors for n2 in neighbors if n1 < n2 and G.has_edge(n1, n2))
    
    if total_possible_connections == 0:
        quality_score = 0
        explanation = "The node has no or only one connection."
    else:
        quality_score = actual_connections / total_possible_connections
        if quality_score < 0.33:
            explanation = "The connections are mostly isolated from each other."
        elif quality_score < 0.67:
            explanation = "There is a moderate level of interconnection among the node's connections."
        else:
            explanation = "The node's connections form a tightly-knit group."
    
    return quality_score, explanation

def generate_insights(G, query, relevant_nodes):
    insights = f"Based on the query '{query}', here are some insights about the relevant nodes:\n\n"
    
    for node in relevant_nodes:
        degree = G.degree(node)
        pagerank = nx.pagerank(G)[node]
        neighbors = list(G.neighbors(node))
        insights += f"- {node}:\n"
        insights += f"  Degree: {degree}\n"
        insights += f"  PageRank: {pagerank:.6f}\n"
        insights += f"  Neighbors: {', '.join(neighbors)}\n"
        insights += f"  Community: {nx.community.louvain_communities(G.subgraph(neighbors))}\n\n"
    
    return insights
