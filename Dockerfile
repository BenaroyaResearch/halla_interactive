# Use Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Install standard Linux build compilers (gcc) required for Python C-extensions (like jenkspy)
RUN apt-get update && apt-get install -y build-essential

# Install R, the specific R packages, Python 3.9, and rpy2 via conda-forge.
RUN conda install -c conda-forge -y \
    python=3.9 \
    r-base=4.3 \
    r-eva \
    r-xicor \
    rpy2

# Install HAllA, Streamlit, and pin Pillow to prevent the Matplotlib/PDF crash
# Added kaleido for static image export (PNG/PDF)
RUN pip install --no-cache-dir \
    halla \
    "Pillow<10.0.0" \
    streamlit \
    plotly \
    pandas \
    kaleido

# Expose the port Streamlit uses to communicate with your browser
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
