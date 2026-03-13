import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import os
import tempfile
from PIL import Image
import scipy.cluster.hierarchy as sch

# --- Helper Functions for HAllA Replication ---

def get_linkage_order(linkage_path, features):
    """Load HAllA linkage matrices and return the pre_order feature list."""
    if linkage_path and os.path.exists(linkage_path):
        try:
            linkage = np.load(linkage_path)
            tree = sch.to_tree(linkage)
            order = tree.pre_order()
            # Map indices back to feature names
            return [features[i] for i in order]
        except Exception as e:
            st.error(f"Error loading linkage: {e}")
    return sorted(features)

def parse_performance_alpha(perf_path):
    """Extract the FDR alpha threshold used in the HAllA run."""
    if os.path.exists(perf_path):
        with open(perf_path, 'r') as f:
            for line in f:
                if 'fdr alpha' in line:
                    try:
                        return float(line.split(':')[-1].strip())
                    except: pass
    return 0.05

def get_block_styling(sig_df, x_order, y_order):
    """Calculate boundaries and labels for significant blocks."""
    shapes = []
    annotations = []
    if sig_df is not None:
        x_map = {feat: i for i, feat in enumerate(x_order)}
        y_map = {feat: i for i, feat in enumerate(y_order)}
        
        for _, row in sig_df.iterrows():
            x_members = [f for f in row['cluster_X'].split(';') if f in x_map]
            y_members = [f for f in row['cluster_Y'].split(';') if f in y_map]
            
            if x_members and y_members:
                x_idxs = [x_map[f] for f in x_members]
                y_idxs = [y_map[f] for f in y_members]
                
                # Bounding box coordinates (cell edges)
                x0, x1 = min(y_idxs) - 0.5, max(y_idxs) + 0.5
                y0, y1 = min(x_idxs) - 0.5, max(x_idxs) + 0.5
                
                shapes.append(dict(
                    type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color="black", width=2.5), xref='x', yref='y'
                ))
                
                # Rank Label
                annotations.append(dict(
                    x=np.mean(y_idxs), y=np.mean(x_idxs),
                    text=f"<b>{row['cluster_rank']}</b>",
                    showarrow=False,
                    font=dict(color="white", size=15),
                    bgcolor="rgba(0,0,0,0.4)",
                    bordercolor="black", borderwidth=1,
                    opacity=0.9
                ))
    return shapes, annotations

# --- Page Configuration ---
st.set_page_config(page_title="HAllA Interactive", layout="wide")

st.title("HAllA: Hierarchical All-against-All Association Testing")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Input Data")
    x_file = st.file_uploader("Upload X Dataset", type=["txt", "tsv"])
    y_file = st.file_uploader("Upload Y Dataset", type=["txt", "tsv"])
    
    st.header("2. HAllA Settings")
    metric = st.selectbox("Correlation Metric", ["spearman", "pearson", "xicor"])
    fdr_alpha_input = st.number_input("FDR Alpha", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    
    st.header("3. Plotting Settings")
    color_scheme = st.selectbox("Heatmap Color Scheme", ['RdBu', 'RdBu_r', 'Viridis', 'Plasma'], index=0)
    show_dots = st.checkbox("Significance Dots (q < Alpha)", value=True)
    show_blocks = st.checkbox("Cluster Block Outlines", value=True)
    trim_plot = st.checkbox("Trim Insignificant Features", value=False)
    
    run_button = st.button("Run HAllA", type="primary")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Main Execution ---
if run_button:
    with st.spinner("Running HAllA..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare files
            x_path = os.path.join(temp_dir, "x.txt")
            y_path = os.path.join(temp_dir, "y.txt")
            if x_file:
                with open(x_path, "wb") as f: f.write(x_file.getbuffer())
            else:
                x_path = "test_data/X_16_100.txt"
            if y_file:
                with open(y_path, "wb") as f: f.write(y_file.getbuffer())
            else:
                y_path = "test_data/Y_16_100.txt"
                
            out_dir = os.path.join(temp_dir, "output")
            cmd = ["halla", "-x", x_path, "-y", y_path, "-m", metric, "-o", out_dir, "--fdr_alpha", str(fdr_alpha_input), "--hallagram", "--plot_file_type", "png"]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Load all generated assets
                res = {}
                res['df'] = pd.read_csv(os.path.join(out_dir, "all_associations.txt"), sep='\t')
                res['sig_df'] = pd.read_csv(os.path.join(out_dir, "sig_clusters.txt"), sep='\t')
                res['alpha'] = parse_performance_alpha(os.path.join(out_dir, "performance.txt"))
                
                # Load feature order from linkage
                x_feat_orig = pd.read_table(os.path.join(out_dir, "X.tsv"), index_col=0).index.tolist()
                y_feat_orig = pd.read_table(os.path.join(out_dir, "Y.tsv"), index_col=0).index.tolist()
                res['x_order'] = get_linkage_order(os.path.join(out_dir, "X_linkage.npy"), x_feat_orig)
                res['y_order'] = get_linkage_order(os.path.join(out_dir, "Y_linkage.npy"), y_feat_orig)
                
                # Load reference image
                hallagram_path = os.path.join(out_dir, "hallagram.png")
                if os.path.exists(hallagram_path):
                    res['ref_img'] = Image.open(hallagram_path)
                
                st.session_state.results = res
                st.success("HAllA computation completed!")
            except Exception as e:
                st.error(f"Execution Error: {e}")

# --- Display Results ---
if st.session_state.results:
    res = st.session_state.results
    df = res['df']
    sig_df = res['sig_df']
    x_order, y_order = res['x_order'], res['y_order']
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Hallagram (Native Matplotlib)")
        if 'ref_img' in res:
            st.image(res['ref_img'], use_container_width=True)
        else:
            st.info("Static image not generated.")

    with col2:
        st.subheader("Plotly Recreation (Interactive)")
        
        # Apply trimming if requested
        if trim_plot and sig_df is not None:
            sig_x = set(';'.join(sig_df['cluster_X']).split(';'))
            sig_y = set(';'.join(sig_df['cluster_Y']).split(';'))
            active_x = [f for f in x_order if f in sig_x]
            active_y = [f for f in y_order if f in sig_y]
        else:
            active_x, active_y = x_order, y_order
            
        pivot_df = df.pivot(index='X_features', columns='Y_features', values='association')
        pivot_df = pivot_df.reindex(index=active_x, columns=active_y)
        
        # Symmetric range centered at 0
        z_abs_max = max(abs(pivot_df.min().min()), abs(pivot_df.max().max()), 0.1)
        
        fig = px.imshow(
            pivot_df,
            color_continuous_scale=color_scheme,
            zmin=-z_abs_max, zmax=z_abs_max,
            labels=dict(x="Y Dataset", y="X Dataset", color="Association"),
            aspect="auto"
        )
        
        # Add HAllA Styling Elements
        shapes, annotations = get_block_styling(sig_df, active_x, active_y)
        fig.update_layout(shapes=shapes, annotations=annotations)
        
        if show_dots:
            sig_pairs = df[(df['q-values'] < res['alpha']) & 
                           (df['X_features'].isin(active_x)) & 
                           (df['Y_features'].isin(active_y))]
            if not sig_pairs.empty:
                fig.add_trace(go.Scatter(
                    x=sig_pairs['Y_features'], y=sig_pairs['X_features'],
                    mode='markers',
                    marker=dict(symbol='circle', color='white', line=dict(color='black', width=1), size=7),
                    name=f"Sig (q < {res['alpha']})"
                ))
        
        fig.update_layout(
            height=650,
            yaxis=dict(side="right", autorange="reversed"), # Right labels, Match HAllA vertical orientation
            xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(
                title="Assoc",
                x=-0.2, # Position on the left side
                xanchor='right',
                len=0.7
            ),
            margin=dict(l=100) # Give space for colorbar on left
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Detail")
    st.dataframe(df.sort_values('q-values').head(20), use_container_width=True)
