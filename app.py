import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from streamlit_plotly_events import plotly_events

from src.ticker_lists import nifty_100_tickers, nifty_50_tickers
# Page configuration
st.set_page_config(page_title="Stock Market Network Analysis", layout="wide")

# Reduce top padding and tighten title spacing for a more compact layout
st.markdown(
        """
        <style>
            .block-container{padding-top:1.0rem;}
            .stTitle h1{margin-top:0rem; margin-bottom:0.12rem;}
        </style>
        """,
        unsafe_allow_html=True,
)

st.title("Graph Structures in Markets")
st.markdown("Interactive 3D analysis of stock correlation networks")

# Load data and precompute expensive operations once
@st.cache_data
def load_data():
    df = pd.read_csv('src/data/india/data_clean_interpolated.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def get_log_returns(_df):
    return np.log(_df.drop(columns=['date'])).diff()

@st.cache_data
def get_available_years(_df):
    return sorted(_df['date'].dt.year.unique().tolist())


def get_corr_matrix(_df, _year, _index="all"):
    year_data = _df[_df['date'].dt.year == _year]
    log_returns_year = np.log(year_data.drop(columns=['date'])).diff()
    
    if _index == "nifty_100":
        available_tickers = [t for t in nifty_100_tickers if t in log_returns_year.columns]
        log_returns_year = log_returns_year[available_tickers]
    elif _index == "nifty_50":
        available_tickers = [t for t in nifty_50_tickers if t in log_returns_year.columns]
        log_returns_year = log_returns_year[available_tickers]
    
    corr = log_returns_year.corr()
    return corr

def get_layout(_edge_keys, _node_list):
    G = nx.Graph()
    G.add_nodes_from(_node_list)
    G.add_edges_from(_edge_keys)
    return nx.spring_layout(G, dim=3, iterations=50, seed=42)

df = load_data()
log_returns = get_log_returns(df)
available_years = get_available_years(df)

# Sidebar controls
st.sidebar.header("Analysis Parameters")
selected_year = st.sidebar.slider(
    "Year", min_value=min(available_years), max_value=max(available_years), 
    value=max(available_years), step=1
)

index_option = st.sidebar.radio(
    "Stock Index",
    ["All stocks", "NIFTY-100", "NIFTY-50"],
    horizontal=True
)

if index_option == "NIFTY-100":
    index_id = "nifty_100"
elif index_option == "NIFTY-50":
    index_id = "nifty_50"
else:
    index_id = "all"

corr_matrix = get_corr_matrix(df, selected_year, index_id)

analysis_type = st.sidebar.radio("Select Analysis",
    ["Correlation Network", "Statistics"])

if analysis_type == "Correlation Network":
    st.sidebar.subheader("Network Settings")
    correlation_threshold = st.sidebar.slider(
        "Correlation Threshold", 0.0, 1.0, 0.25, 0.05
    )
    st.sidebar.caption(f"Year: {selected_year} | Index: {index_option}")

    keys = tuple(corr_matrix.columns.tolist())
    
    # Statistics at top (compact) — include Year and Threshold
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,0.8,0.8])

    # Vectorized edge building — no nested Python loops
    corr_pairs = corr_matrix.stack()
    corr_pairs = corr_pairs[
        corr_pairs.index.get_level_values(0) < corr_pairs.index.get_level_values(1)
    ]
    filtered = corr_pairs[corr_pairs >= correlation_threshold]
    edge_keys = tuple((u, v) for u, v in filtered.index)
    # Linear interpolation: threshold→distance=2, max_corr(1.0)→distance=1
    edge_distances = {(u, v): float(1.0 + (1.0 - corr_val) / (1.0 - correlation_threshold)) for (u, v), corr_val in filtered.items()}

    G_corr = nx.Graph()
    G_corr.add_nodes_from(keys)
    for (u, v), dist in edge_distances.items():
        G_corr.add_edge(u, v, distance=dist)

    # Display compact statistics (smaller text)
    with col1:
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Nodes</strong><br><span style='font-size:18px'>{G_corr.number_of_nodes()}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Edges</strong><br><span style='font-size:18px'>{G_corr.number_of_edges()}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Density</strong><br><span style='font-size:18px'>{nx.density(G_corr):.4f}</span></div>", unsafe_allow_html=True)
    with col4:
        degrees = [G_corr.degree(node) for node in G_corr.nodes()]
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Avg Degree</strong><br><span style='font-size:18px'>{np.mean(degrees):.2f}</span></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Year</strong><br><span style='font-size:18px'>{selected_year}</span></div>", unsafe_allow_html=True)
    with col6:
        st.markdown(f"<div style='font-size:12px;margin-bottom:2px'><strong>Threshold</strong><br><span style='font-size:18px'>{correlation_threshold:.2f}</span></div>", unsafe_allow_html=True)
    
    st.caption("💡 **Interactions**: Scroll to zoom | Click & drag to rotate | Hover over nodes to see ticker")

    # Layout keyed by (year, index) only — threshold changes never trigger a recompute
    layout_key = (selected_year, index_id)
    cached = st.session_state.get("layout_cache", {})
    if layout_key in cached and set(cached[layout_key].keys()) == set(keys):
        pos_3d = cached[layout_key]
    else:
        prev_pos = st.session_state.get("pos_3d", None)
        init_pos = prev_pos if (prev_pos is not None and set(prev_pos.keys()) == set(keys)) else None
        pos_3d = nx.spring_layout(G_corr, dim=3, iterations=30, seed=42, pos=init_pos)
        cached[layout_key] = pos_3d
        st.session_state["layout_cache"] = cached
    st.session_state["pos_3d"] = pos_3d
    
    node_x = [pos_3d[node][0] for node in G_corr.nodes()]
    node_y = [pos_3d[node][1] for node in G_corr.nodes()]
    node_z = [pos_3d[node][2] for node in G_corr.nodes()]
    node_labels = list(G_corr.nodes())
    
    edge_x, edge_y, edge_z = [], [], []
    for edge in G_corr.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, 
        mode='lines', 
        line=dict(color='gray', width=2), 
        showlegend=False,
        hoverinfo='none'
    )
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, 
        mode='markers+text',
        marker=dict(size=6, color='lightblue', opacity=0.8, line=dict(color='darkblue', width=1)),
        text=node_labels, 
        textposition='top center',
        textfont=dict(size=8),
        hoverinfo='text',
        hovertext=node_labels,
        showlegend=False
    )
    
    # Restore camera from session state
    if "camera" not in st.session_state:
        st.session_state.camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        uirevision="stable",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=12, l=5, r=5, t=18),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            camera=st.session_state.camera
        ),
        width=800,
        height=600
    )
    
    # Capture events and update camera from relayout data (smaller canvas)
    selected_points = plotly_events(fig, override_height=600, override_width=800, key="corr_graph")
    
    # Extract camera from relayout event if present
    if selected_points and isinstance(selected_points, dict) and "scene.camera" in selected_points:
        st.session_state.camera = selected_points["scene.camera"]

elif analysis_type == "Statistics":
    st.subheader(f"Correlation Analysis ({selected_year}) - {index_option}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Correlation Matrix Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix.iloc[:20, :20], annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("**Correlation Distribution**")
        corr_flat = corr_matrix.values[np.tril_indices_from(corr_matrix.values, k=-1)]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(corr_flat, bins=30, edgecolor='k', density=True, alpha=0.7)
        
        mu, sigma = norm.fit(corr_flat)
        x = np.linspace(corr_flat.min(), corr_flat.max(), 100)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
        
        ax.set_title('Histogram of Correlation Coefficients')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    st.write("**Data Summary**")
    year_data = df[df['date'].dt.year == selected_year]
    corr_values = corr_matrix.values[np.tril_indices_from(corr_matrix.values, k=-1)]
    mean_corr = corr_values.mean()
    st.write(f"- Year: {selected_year}")
    st.write(f"- Number of stocks: {len(corr_matrix.columns)}")
    st.write(f"- Trading days: {len(year_data)}")
    st.write(f"- Mean correlation: {mean_corr:.4f}")
    with st.expander("Debug Info"):
        st.write(f"Min: {corr_values.min():.4f}, Max: {corr_values.max():.4f}")
        st.write(f"Std Dev: {corr_values.std():.4f}")
