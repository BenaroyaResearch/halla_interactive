import streamlit as st
import subprocess
import os
import tempfile
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="HAllA Interactive", layout="wide")

st.title("HAllA: Hierarchical All-against-All Association Testing")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Input Data")
    st.write("Upload tab-delimited text files. If left blank, the app will use default toy data.")
    x_file = st.file_uploader("Upload X Dataset", type=["txt", "tsv"])
    y_file = st.file_uploader("Upload Y Dataset", type=["txt", "tsv"])
    
    st.header("2. HAllA Settings")
    metric = st.selectbox("Correlation Metric", ["spearman", "pearson", "xicor"])
    fdr_alpha = st.number_input("FDR Alpha", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    
    run_button = st.button("Run HAllA", type="primary")

# --- Main Execution Area ---
if run_button:
    with st.spinner("Running HAllA... This may take a few minutes."):
        
        # We must use a temporary directory because HAllA requires physical files on disk, 
        # but Streamlit file uploads are temporarily held in memory.
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Handle X Dataset (Uploaded vs Default)
            if x_file:
                x_path = os.path.join(temp_dir, "x_input.txt")
                with open(x_path, "wb") as f:
                    f.write(x_file.getbuffer())
            else:
                x_path = "test_data/X_16_100.txt"
                
            # Handle Y Dataset (Uploaded vs Default)
            if y_file:
                y_path = os.path.join(temp_dir, "y_input.txt")
                with open(y_path, "wb") as f:
                    f.write(y_file.getbuffer())
            else:
                y_path = "test_data/Y_16_100.txt"
                
            out_dir = os.path.join(temp_dir, "halla_output")
            
            # Construct the HAllA command (mirroring your Bash script)
            cmd = [
                "halla",
                "-x", x_path,
                "-y", y_path,
                "-m", metric,
                "-o", out_dir,
                "--fdr_alpha", str(fdr_alpha),
                "--plot_file_type", "png",
                "--clustermap",
                "--diagnostic_plot"
            ]
            
            try:
                # Execute HAllA in the background
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                st.success("HAllA computation completed successfully!")
                
                # --- GEMINI CLI GENERATED STEP: Print the Hallagram ---
                hallagram_path = os.path.join(out_dir, "hallagram.png")
                
                if os.path.exists(hallagram_path):
                    st.subheader("HAllA Clustergram")
                    image = Image.open(hallagram_path)
                    st.image(image, caption=f"Static Hallagram (FDR: {fdr_alpha})", use_container_width=True)
                else:
                    st.warning("HAllA ran successfully, but 'hallagram.png' was not generated. This usually means no significant associations were found at the selected FDR threshold.")
                # ------------------------------------------------------
                
            except subprocess.CalledProcessError as e:
                st.error("HAllA Execution Failed.")
                st.code(e.stderr)
