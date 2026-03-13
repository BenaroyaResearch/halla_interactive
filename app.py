import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import os
import tempfile

# --- Helper Functions for HAllA Replication ---

def reorder_features(features, clusters_str_list):
    """Reorder features to ensure clusters are contiguous for rectangular blocks."""
    ordered = []
    # Add features from significant clusters first
    for c_str in clusters_str_list:
        for f in c_str.split(';'):
            if f in features and f not in ordered:
                ordered.append(f)
    # Add any remaining features
    for f in features:
        if f not in ordered:
            ordered.append(f)
    return ordered

def parse_clusters(sig_df, x_order, y_order):
    """Parse clusters into coordinate boundaries for Plotly shapes."""
    blocks = []
    if sig_df is not None:
        x_map = {feat: i for i, feat in enumerate(x_order)}
        y_map = {feat: i for i, feat in enumerate(y_order)}
        
        for _, row in sig_df.iterrows():
            x_feats = row['cluster_X'].split(';')
            y_feats = row['cluster_Y'].split(';')
            
            # Find indices in the current order
            x_idxs = [x_map[f] for f in x_feats if f in x_map]
            y_idxs = [y_map[f] for f in y_feats if f in y_map]
            
            if x_idxs and y_idxs:
                blocks.append({
                    'rank': row['cluster_rank'],
                    'x0': min(y_idxs) - 0.5,
                    'x1': max(y_idxs) + 0.5,
                    'y0': min(x_idxs) - 0.5,
                    'y1': max(x_idxs) + 0.5,
                    'xc': np.mean(y_idxs),
                    'yc': np.mean(x_idxs)
                })
    return blocks

# --- Page Configuration ---
st.set_page_config(page_title="HAllA Interactive", layout="wide")

st.title("HAllA: Hierarchical All-against-All Association Testing")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Input Data")
    x_file = st.file_uploader("Upload X Dataset", type=["txt", "tsv"])
    y_file = st.file_uploader("Upload Y Dataset", type=["txt", "tsv"])
    
    st.header("2. HAllA Settings")
    metric = st.selectbox("Correlation Metric", ["spearman", "pearson", "xicor"])
    fdr_alpha = st.number_input("FDR Alpha", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    
    st.header("3. Plotting Settings")
    color_scheme = st.selectbox(
        "Heatmap Color Scheme", 
        ['RdBu', 'Viridis', 'Plasma', 'Inferno', 'Cividis', 'Spectral'], 
        index=0
    )
    
    st.header("4. HAllA Style Replication")
    show_dots = st.checkbox("Significance Dots (q < Alpha)", value=True)
    show_blocks = st.checkbox("Cluster Blocks (Outlines)", value=True)
    show_ranks = st.checkbox("Rank Numbers", value=True)
    trim_plot = st.checkbox("Trim Insignificant Features", value=False)
    
    run_button = st.button("Run HAllA", type="primary")

# Initialize session state
if 'halla_df' not in st.session_state:
    st.session_state.halla_df = None
if 'sig_df' not in st.session_state:
    st.session_state.sig_df = None
if 'last_alpha' not in st.session_state:
    st.session_state.last_alpha = 0.05

# --- Main Execution Area ---
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
            cmd = ["halla", "-x", x_path, "-y", y_path, "-m", metric, "-o", out_dir, "--fdr_alpha", str(fdr_alpha), "--no_hallagram"]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                assoc_path = os.path.join(out_dir, "all_associations.txt")
                sig_path = os.path.join(out_dir, "sig_clusters.txt")
                
                if os.path.exists(assoc_path):
                    st.session_state.halla_df = pd.read_csv(assoc_path, sep='\t')
                    st.session_state.sig_df = pd.read_csv(sig_path, sep='\t') if os.path.exists(sig_path) else None
                    st.session_state.last_alpha = fdr_alpha
                    st.success("HAllA computation completed successfully!")
                else:
                    st.error("Results not found.")
            except subprocess.CalledProcessError as e:
                st.error(f"HAllA Failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")

# --- Results Rendering ---
if st.session_state.halla_df is not None:
    df = st.session_state.halla_df.copy()
    sig_df = st.session_state.sig_df
    alpha = st.session_state.last_alpha
    
    # Apply Trimming if requested
    if trim_plot and sig_df is not None:
        sig_x = set(';'.join(sig_df['cluster_X']).split(';'))
        sig_y = set(';'.join(sig_df['cluster_Y']).split(';'))
        df = df[df['X_features'].isin(sig_x) & df['Y_features'].isin(sig_y)]

    if not df.empty:
        # Reorder to ensure clusters are rectangular
        x_ordered = reorder_features(df['X_features'].unique().tolist(), sig_df['cluster_X'].tolist() if sig_df is not None else [])
        y_ordered = reorder_features(df['Y_features'].unique().tolist(), sig_df['cluster_Y'].tolist() if sig_df is not None else [])
        
        pivot_df = df.pivot(index='X_features', columns='Y_features', values='association')
        pivot_df = pivot_df.reindex(index=x_ordered, columns=y_ordered)

        # Plotly Figure
        # Center colorbar at 0 to match HAllA style
        limit = max(abs(pivot_df.min().min()), abs(pivot_df.max().max()), 0.1)
        
        fig = px.imshow(
            pivot_df,
            color_continuous_scale=color_scheme,
            zmin=-limit, zmax=limit,
            labels=dict(color="Assoc", x="Y Features", y="X Features"),
            aspect="auto",
            title="HAllA Interactive Hallagram"
        )
        
        # 1. Block Highlighting (Replicating HAllA Black Borders)
        if show_blocks and sig_df is not None:
            blocks = parse_clusters(sig_df, x_ordered, y_ordered)
            for b in blocks:
                fig.add_shape(
                    type="rect", x0=b['x0'], y0=b['y0'], x1=b['x1'], y1=b['y1'],
                    line=dict(color="black", width=2), xref='x', yref='y'
                )
                # 2. Rank Numbers (Replicating HAllA Centered Ranks)
                if show_ranks:
                    fig.add_annotation(
                        x=b['xc'], y=b['yc'], text=str(b['rank']),
                        showarrow=False, font=dict(color="white", size=14, family="Arial Black"),
                        bgcolor="rgba(0,0,0,0.5)", bordercolor="black", borderwidth=1
                    )

        # 3. Significance Dots (Replicating HAllA Dots for q < alpha)
        if show_dots:
            sig_pairs = df[df['q-values'] < alpha]
            if not sig_pairs.empty:
                fig.add_trace(go.Scatter(
                    x=sig_pairs['Y_features'],
                    y=sig_pairs['X_features'],
                    mode='markers',
                    marker=dict(symbol='circle', color='white', line=dict(color='black', width=1), size=6),
                    name=f'Sig (q < {alpha})',
                    showlegend=True
                ))

        fig.update_layout(
            height=700,
            coloraxis_colorbar=dict(title="Assoc", x=1.02),
            yaxis=dict(side="right") # Match HAllA labels on right
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display Data Table
        st.subheader("Significant Associations (Sorted by q-value)")
        st.dataframe(df.sort_values('q-values').head(20), use_container_width=True)
        
        # Download Option
        csv = df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button("Download All Associations (.txt)", csv, "halla_results.txt", "text/tab-separated-values")
    else:
        st.warning("No data to display. Try a different FDR threshold or datasets.")
