# Enhanced Network Analysis Project

## Project Overview

This project is an advanced interactive network analysis tool designed for educational purposes. It provides a comprehensive suite of features for exploring, visualizing, and analyzing complex network structures. The user-friendly interface, powered by Streamlit, allows users to gain insights into network dynamics, community structures, and individual node characteristics.

## Key Features

1. **Network Visualization**: Interactive graph representation using PyVis, offering a clear view of the network structure.

2. **Basic Statistics**: Quick overview of network properties including node count, edge count, average degree, and connected components.

3. **Concept Explanation**: In-depth explanations of key network analysis concepts to support learning and understanding.

4. **Individual Node Analysis**: Detailed examination of specific nodes, including degree, component membership, and PageRank score.

5. **Influence Paths**: Identification and visualization of shortest paths between nodes, illustrating information or influence flow.

6. **Community Detection**: Implementation of the Louvain algorithm to identify and visualize network communities.

7. **Connection Quality Analysis**: Evaluation of node connections based on the interconnectedness of their neighbors.

8. **RAG Insights**: AI-powered insights leveraging TF-IDF and FAISS for efficient similarity search and information retrieval.

9. **Gamification**: Interactive "what-if" scenarios allowing users to add or remove nodes and observe the impact on network statistics and PageRank scores.

10. **Flexible Reporting**: Generation of comprehensive reports in various formats (CSV, PDF, DOCX) for further analysis and presentation.

## Technology Stack

- **Streamlit**: Web application framework for the user interface
- **NetworkX**: Graph creation and analysis
- **Pandas**: Data manipulation and analysis
- **Plotly & PyVis**: Interactive data visualization
- **FAISS**: Efficient similarity search and clustering
- **scikit-learn**: Machine learning tools, including TF-IDF vectorization
- **FPDF & python-docx**: Report generation in PDF and DOCX formats

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Usage

Navigate through different analysis types using the sidebar. Each section provides unique insights into the network:

- **Overview**: General network statistics and visualization
- **Concept Explanation**: Learn about network analysis terminology
- **Individual Node Analysis**: Explore specific nodes in detail
- **Influence Paths**: Find connections between nodes
- **Community Detection**: Identify clusters within the network
- **Connection Quality**: Assess the strength of node connections
- **RAG Insights**: Get AI-generated insights about the network
- **Gamification**: Experiment with network modifications

## Contributing

Contributions to enhance the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- NetworkX community for their comprehensive graph analysis library
- Streamlit team for their intuitive web application framework
- All contributors and users of this educational tool

