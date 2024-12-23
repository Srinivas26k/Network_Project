# Enhanced Network Analysis Project

## Overview

This Enhanced Network Analysis Project is an interactive, educational tool designed to explore and analyze complex network structures. Built with Python and leveraging state-of-the-art libraries, this application provides a comprehensive suite of network analysis features, from basic statistics to advanced community detection and AI-powered insights.

## Features

- **Interactive Network Visualization**: Explore network structures visually.
- **Basic Network Statistics**: Quick overview of key network properties.
- **Concept Explanations**: Learn about fundamental network analysis concepts.
- **Individual Node Analysis**: Dive deep into specific node characteristics.
- **Influence Path Detection**: Discover shortest paths between nodes.
- **Community Detection**: Identify and visualize network communities.
- **Connection Quality Analysis**: Evaluate the strength of node connections.
- **RAG (Retrieval Augmented Generation) Insights**: AI-powered network analysis.
- **Network Gamification**: Experiment with "what-if" scenarios by modifying the network.
- **Flexible Reporting**: Generate comprehensive reports in various formats.

## Technology Stack

- **Streamlit**: For the interactive web interface
- **NetworkX**: For graph creation and analysis
- **Pandas**: For data manipulation
- **Plotly & PyVis**: For interactive visualizations
- **FAISS**: For efficient similarity search
- **scikit-learn**: For TF-IDF vectorization
- **FPDF & python-docx**: For report generation

### **Installation**

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**

   First, clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/Srinivas26k/Ropar_Network_Project.git
   ```

2. **Navigate to the Project Directory**

   After cloning, navigate to the project directory:
   ```bash
   cd Ropar_Network_Project
   ```

3. **Create and Activate a Virtual Environment (Optional but Recommended)**

   To keep dependencies isolated, it's recommended to create a virtual environment. Run the following command to create and activate the environment:

   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Install the Required Dependencies**

   Install the necessary dependencies for the project by running:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install streamlit networkx pandas plotly pyvis faiss-cpu numpy fpdf python-docx
   ```

6. **Run the Application**

   Once the dependencies are installed, you can run the application using:
   ```bash
   streamlit run app.py
   ```

---



