import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pyvis.network import Network
import streamlit.components.v1 as components
from sympy import symbols, Eq, solve

# Set up page configuration
st.set_page_config(page_title="Network Analysis Tool", layout="wide")

# Helper function to visualize clustering coefficients
def visualize_clustering_coefficients(G):
    pos = nx.spring_layout(G)
    clustering_coeffs = nx.clustering(G)
    
    node_colors = [clustering_coeffs[node] for node in G.nodes()]
    cmap = plt.cm.viridis

    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, 
        node_color=node_colors, 
        cmap=cmap, 
        node_size=500, 
        edge_color='gray'
    )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                                norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, label="Clustering Coefficient")
    plt.title("Visualization of Clustering Coefficients")
    st.pyplot(plt)

# Enhanced clustering coefficient visualization with seaborn
def clustering_coefficient_distribution(G):
    clustering_coeffs = list(nx.clustering(G).values())

    # Plot distribution with seaborn
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(clustering_coeffs, kde=True, ax=ax[0], color="blue")
    ax[0].set_title("Clustering Coefficient Distribution")
    ax[0].set_xlabel("Clustering Coefficient")
    ax[0].set_ylabel("Frequency")

    sns.boxplot(x=clustering_coeffs, ax=ax[1], color="green")
    ax[1].set_title("Clustering Coefficient Boxplot")
    ax[1].set_xlabel("Clustering Coefficient")

    st.pyplot(fig)

    # Fit normal distribution to the data
    mu, std = norm.fit(clustering_coeffs)
    st.write(f"Fitted Normal Distribution: Mean = {mu:.4f}, Std Dev = {std:.4f}")

# Sidebar: File Upload
st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload your network data (CSV):", type=["csv"])

# Main Content
st.title("Network Analysis with Clustering")
if data_file:
    # Load data and create graph
    try:
        data = pd.read_csv(data_file)
        if len(data.columns) < 2:
            st.error("Please upload a valid edge list with at least two columns.")
        else:
            G = nx.from_pandas_edgelist(data, source=data.columns[0], target=data.columns[1])
            
            # Display basic graph metrics
            st.subheader("Graph Metrics")
            st.write(f"Number of Nodes: {G.number_of_nodes()}")
            st.write(f"Number of Edges: {G.number_of_edges()}")

            avg_clustering = nx.average_clustering(G)
            st.write(f"Average Clustering Coefficient: {avg_clustering:.4f}")

            # Clustering Analysis Section
            st.subheader("Clustering Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Clustering Coefficient Metrics")
                clustering_coeffs = nx.clustering(G)
                for node, coeff in clustering_coeffs.items():
                    st.write(f"Node {node}: Clustering Coefficient = {coeff:.4f}")

                st.write(f"Average Clustering Coefficient: {avg_clustering:.4f}")

            with col2:
                st.subheader("Clustering Coefficient Distribution")
                clustering_coefficient_distribution(G)

            # Visualize clustering with enhanced layout
            st.subheader("Visualization of Clustering Coefficients")
            visualize_clustering_coefficients(G)

            # Interactive network visualization with PyVis
            st.subheader("Interactive Network Visualization")
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)

            for node, coeff in nx.clustering(G).items():
                net.get_node(node)["title"] = f"Clustering Coefficient: {coeff:.4f}"
                net.get_node(node)["color"] = plt.cm.viridis(coeff)

            net.save_graph("network.html")
            HtmlFile = open("network.html", 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=600)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a network edge list in CSV format.")
