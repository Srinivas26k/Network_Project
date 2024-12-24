import networkx as nx
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df['Name'] = df['Name'].astype(str).str.strip()
    df['Choose your pick'] = df['Choose your pick'].astype(str).str.strip()
    return df

def create_network(df):
    G = nx.Graph()
    G.add_nodes_from(df['Name'].unique())
    G.add_edges_from(zip(df['Name'], df['Choose your pick']))
    return G

def create_embeddings(names, vectorizer):
    return vectorizer.fit_transform(names)

def query_embeddings(query, embeddings, names, vectorizer, k=5):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [names[i] for i in top_indices if i < len(names)]

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
        insights += f"  Community: {next(community for community in nx.community.greedy_modularity_communities(G) if node in community)}\n\n"
    return insights

def calculate_connection_quality(G, node):
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

def common_neighbors_score(G, node1, node2):
    return len(list(nx.common_neighbors(G, node1, node2)))

def predict_links(G, top_k=10):
    non_edges = list(nx.non_edges(G))
    scores = [(u, v, common_neighbors_score(G, u, v)) for u, v in non_edges]
    return sorted(scores, key=lambda x: x[2], reverse=True)[:top_k]

def calculate_network_metrics(G):
    return {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Average Degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "Number of Connected Components": nx.number_connected_components(G),
        "Average Clustering Coefficient": nx.average_clustering(G),
        "Network Density": nx.density(G)
    }

