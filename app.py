import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import os
import tempfile

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
    
    st.header("3. Plotting Settings")
    color_scheme = st.selectbox(
        "Heatmap Color Scheme", 
        ['RdBu', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'RdYlBu', 'Spectral'], 
        index=0
    )
    
    run_button = st.button("Run HAllA", type="primary")

# Initialize session state to persist results across re-runs (like color scheme changes)
if 'halla_df' not in st.session_state:
    st.session_state.halla_df = None
if 'last_metric' not in st.session_state:
    st.session_state.last_metric = None

# --- Main Execution Area ---
if run_button:
    with st.spinner("Running HAllA... This may take a few minutes."):
        
        # We must use a temporary directory because HAllA requires physical files on disk
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
            
            # Construct the HAllA command
            cmd = [
                "halla",
                "-x", x_path,
                "-y", y_path,
                "-m", metric,
                "-o", out_dir,
                "--fdr_alpha", str(fdr_alpha),
                "--no_hallagram"  # Replacing static plot with Plotly
            ]
            
            try:
                # Execute HAllA
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # HAllA typically outputs 'all_associations.txt'
                # We also check for 'results.txt' and 'similarity_matrix.txt' as requested
                results_file = None
                for candidate in ["all_associations.txt", "results.txt", "similarity_matrix.txt"]:
                    path = os.path.join(out_dir, candidate)
                    if os.path.exists(path):
                        results_file = path
                        break
                
                if results_file:
                    df = pd.read_csv(results_file, sep='\t')
                    st.session_state.halla_df = df
                    st.session_state.last_metric = metric
                    st.success("HAllA computation completed successfully!")
                else:
                    st.error("HAllA ran but results file not found in output directory.")
                    st.session_state.halla_df = None
                
            except subprocess.CalledProcessError as e:
                st.error("HAllA Execution Failed.")
                st.code(e.stderr)
                st.session_state.halla_df = None

# --- Results Rendering ---
if st.session_state.halla_df is not None:
    df = st.session_state.halla_df
    
    # 1. Interactive Heatmap (Requirement: Parse similarity_matrix.txt and use px.imshow)
    st.subheader("Interactive Association Heatmap")
    
    try:
        # Pivot long-form associations into a matrix for imshow
        # Columns in HAllA 0.8.x: X_features, Y_features, association
        pivot_df = df.pivot(index='X_features', columns='Y_features', values='association')
        
        fig = px.imshow(
            pivot_df,
            color_continuous_scale=color_scheme,
            labels=dict(color="Association"),
            aspect="auto",
            title=f"HAllA Pairwise Associations ({st.session_state.last_metric})"
        )
        
        # Plotly puts color bar on the right by default.
        # We ensure it stays there and the layout is clean.
        fig.update_layout(coloraxis_colorbar=dict(title="Assoc", x=1.02))
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
        st.write("Data Preview:", df.head())

    # 2. Stats Output (Requirement: Parse results.txt and display top 20 rows)
    st.subheader("Top 20 Associations")
    
    # Sort by q-values if available (standard HAllA output)
    if 'q-values' in df.columns:
        display_df = df.sort_values('q-values').head(20)
    elif 'p-values' in df.columns:
        display_df = df.sort_values('p-values').head(20)
    else:
        display_df = df.head(20)
        
    st.dataframe(display_df, use_container_width=True)
