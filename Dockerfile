# Use Miniconda base image to easily manage R and Python dependencies
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Install R, the specific R packages, Python 3.9, and rpy2 via conda-forge.
# Installing rpy2 via conda-forge on Linux ensures perfect linkage between 
# Conda's R and Python environments without manual C-flag compilation.
RUN conda install -c conda-forge -y \
    python=3.9 \
    r-base=4.3 \
    r-eva \
    r-xicor \
    rpy2

# Install HAllA, Streamlit, and pin Pillow to prevent the Matplotlib/PDF crash
RUN pip install --no-cache-dir \
    halla \
    "Pillow<10.0.0" \
    streamlit \
    plotly \
    pandas

# Expose the port Streamlit uses to communicate with your browser
EXPOSE 8501

# Command to run the Streamlit app (placeholder for now)
CMD ["streamlit", "hello"]
