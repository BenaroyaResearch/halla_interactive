# HAllA Interactive 

An interactive, containerized web application for visualizing and exploring Hierarchical All-against-All (HAllA) (Ghazi et. al, Bioinformatics, 2022)association testing results. 

This app provides a streamlined Streamlit interface to execute HAllA, dynamically explore the resulting similarity matrices using Plotly, zoom in on significant feature blocks, and export publication-ready PDFs.

## Prerequisites
To ensure complete reproducibility and avoid dependency conflicts across different operating systems, this application is fully containerized. 
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running on your machine.

## Quick Start (Local Build)

**1. Clone the repository**
```bash
git clone [https://github.com/BenaroyaResearch/halla_interactive.git](https://github.com/BenaroyaResearch/halla_interactive.git)
cd halla_interactive
```

**2. Build the Docker Image**
```bash
docker build -t halla_interactive:latest .
```

**3. Run the Application**
```bash
ddocker run --rm -p 8501:8501 -v "$(pwd):/app" halla_interactive:latest streamlit run app.py
```
