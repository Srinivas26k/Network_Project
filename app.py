import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network
import faiss
import numpy as np
from fpdf import FPDF
from docx import Document
import io
import base64
import random

from utils import preprocess_data, create_network, create_embeddings, query_faiss, generate_report

# Set up FAISS
dimension = 768  # Dimension of the nomic-embed-text embeddings
index = faiss.IndexFlatL2(dimension)

st.set_page_config(page_title="Interactive Network Analysis", layout="wide")
st.title("Interactive Network Analysis with AI-Powered Insights")

# Load and preprocess data
@st.cache_data
def load_data():
    # Assuming the CSV file is in the same directory as the script
    df = pd.read_csv("impression_network.csv")
    return preprocess_data(df)

df = load_data()

# Create network graph
G = create_network(df)

# Create embeddings and add to FAISS index
embeddings = create_embeddings(df['Name'].tolist())
index.add(embeddings)

# Sidebar for user input
st.sidebar.header("Network Analysis Options")
analysis_type = st.sidebar.selectbox("Choose Analysis Type", ["Overview", "Influence Paths", "Community Detection", "Simple Insights"])

if analysis_type == "Overview":
    st.subheader("Network Overview")
    
    # Display basic network statistics
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    
    # Create and display network visualization
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.save_graph("network.html")
    st.components.v1.html(open("network.html", "r").read(), height=600)

elif analysis_type == "Influence Paths":
    st.subheader("Influence Paths")
    
    source = st.selectbox("Select source node", df['Name'].tolist())
    target = st.selectbox("Select target node", df['Name'].tolist())
    
    if st.button("Find Influence Path"):
        try:
            path = nx.shortest_path(G, source=source, target=target)
            st.write(f"Influence path from {source} to {target}:")
            st.write(" -> ".join(path))
        except nx.NetworkXNoPath:
            st.write(f"No path found between {source} and {target}")

elif analysis_type == "Community Detection":
    st.subheader("Community Detection")
    
    communities = nx.community.louvain_communities(G)
    
    st.write(f"Number of communities detected: {len(communities)}")
    
    # Visualize communities
    pos = nx.spring_layout(G)
    fig = go.Figure()
    
    for i, community in enumerate(communities):
        node_x = [pos[node][0] for node in community]
        node_y = [pos[node][1] for node in community]
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', name=f'Community {i+1}',
                                 marker=dict(size=10, line=dict(width=2))))
    
    fig.update_layout(title="Community Detection Visualization", showlegend=True)
    st.plotly_chart(fig)

elif analysis_type == "Simple Insights":
    st.subheader("Simple Network Insights")
    
    # Generate simple insights about the network
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
    num_connected_components = nx.number_connected_components(G)
    
    st.write(f"Average degree: {avg_degree:.2f}")
    st.write(f"Node with highest degree: {max_degree_node}")
    st.write(f"Number of connected components: {num_connected_components}")
    
    # Simple recommendation
    random_node = random.choice(list(G.nodes()))
    neighbors = list(G.neighbors(random_node))
    if neighbors:
        recommendation = random.choice(neighbors)
        st.write(f"Random recommendation: {random_node} might want to connect with {recommendation}")
    else:
        st.write(f"Random node {random_node} has no connections to recommend")

# Report generation
st.sidebar.header("Generate Report")
report_format = st.sidebar.selectbox("Choose report format", ["CSV", "PDF", "DOCX"])

if st.sidebar.button("Generate Report"):
    report_data = generate_report(G, df, analysis_type)
    
    if report_format == "CSV":
        csv = report_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="network_analysis_report.csv">Download CSV Report</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    elif report_format == "PDF":
        pdf_buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Network Analysis Report", ln=True, align='C')
        for key, value in report_data.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        pdf.output(pdf_buffer)
        b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="network_analysis_report.pdf">Download PDF Report</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    elif report_format == "DOCX":
        doc_buffer = io.BytesIO()
        doc = Document()
        doc.add_heading("Network Analysis Report", 0)
        for key, value in report_data.items():
            doc.add_paragraph(f"{key}: {value}")
        doc.save(doc_buffer)
        b64 = base64.b64encode(doc_buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="network_analysis_report.docx">Download DOCX Report</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Created by Your Name")

