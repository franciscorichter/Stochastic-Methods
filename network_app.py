import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# For consistent styling
np.random.seed(42)
sns.set(style="whitegrid")

def get_color_scheme(bw_mode):
    """Return color parameters depending on black & white mode."""
    if bw_mode:
        return {
            "node_color": "black",
            "edge_color": "black",
            "label_color": "black",
            "kde_color": "black",
            "cmap_heatmap": "Greys"
        }
    else:
        return {
            "node_color": "skyblue",
            "edge_color": "gray",
            "label_color": "darkblue",
            "kde_color": "red",
            "cmap_heatmap": "YlGnBu"
        }

# --------------------- RANDOM NETWORK MODELS ---------------------

def plot_simple_graph(bw_mode):
    G = nx.Graph()
    edges = [(1,2), (2,3), (1,3), (2,4), (3,4), (4,5)]
    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from(edges)

    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(5,4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, 
            node_color=cs["node_color"], 
            edge_color=cs["edge_color"], 
            node_size=600)
    plt.title("Simple Graph with 5 Nodes")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

    # Plot adjacency matrix
    A = nx.adjacency_matrix(G).todense()
    plt.figure(figsize=(5,4))
    sns.heatmap(A, annot=True, cmap=cs["cmap_heatmap"], cbar=True, square=True)
    plt.title("Adjacency Matrix")
    plt.xlabel("Node")
    plt.ylabel("Node")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_erdos_renyi(n, p, bw_mode):
    G = nx.erdos_renyi_graph(n, p, seed=42)
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color=cs["node_color"],
            edge_color=cs["edge_color"],
            node_size=500)
    plt.title(f"Erdős-Rényi Graph (n={n}, p={p})")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

def plot_watts_strogatz(n, k, p, bw_mode):
    G = nx.watts_strogatz_graph(n, k, p, seed=42)
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color=cs["node_color"],
            edge_color=cs["edge_color"],
            node_size=500)
    plt.title(f"Watts-Strogatz Graph (n={n}, k={k}, p={p})")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

def plot_barabasi_albert(n, m, bw_mode):
    G = nx.barabasi_albert_graph(n, m, seed=42)
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color=cs["node_color"],
            edge_color=cs["edge_color"],
            node_size=500)
    plt.title(f"Barabási-Albert Graph (n={n}, m={m})")
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

# --------------------- MONTE CARLO EXAMPLES ---------------------

def plot_monte_carlo_business_partnership(bw_mode):
    """
    Fix for "too many values to unpack": use (u, v, w) for edges(2, data="weight").
    """
    st.write(r"""
    **Business Partnership with Missing Strengths**

    - A node is a business; an edge \((u,v)\) has a partnership strength \(w_{uv}\).
    - One edge is missing, so we sample its weight from Uniform(0.5, 2.5).
    - We compute the weighted degree of node 2 across many samples.
    """)
    G = nx.Graph()
    known_edges = [(1,2,0.8), (1,3,0.5), (2,3,0.7), (3,4,0.6), (4,5,0.9)]
    G.add_nodes_from([1,2,3,4,5])
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)
    # Missing edge: (2,4)
    num_samples = 300
    missing_samples = np.random.uniform(0.5, 2.5, num_samples)
    w_degs = []
    for val in missing_samples:
        G_temp = G.copy()
        G_temp.add_edge(2, 4, weight=val)
        # Weighted degree of node 2:
        wd_2 = 0
        for u, v, w in G_temp.edges(2, data="weight"):
            wd_2 += w
        w_degs.append(wd_2)
    
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,4))
    sns.histplot(w_degs, bins=25, kde=True, color=cs["node_color"])
    plt.axvline(np.mean(w_degs), color=cs["kde_color"], linestyle="dashed", label="Mean")
    plt.title("Weighted Degree of Node 2 (Missing Edge (2,4))")
    plt.xlabel("Weighted Degree")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_monte_carlo_supply_chain(bw_mode):
    """
    Fix for "too many values to unpack": we won't do edges(2, data=...), 
    but we'll do shortest_path_length with an uncertain edge (2,4).
    """
    st.write(r"""
    **Supply Chain with Uncertain Edges**

    - Nodes: companies. 
    - Edges: known supplier–customer links. 
    - One uncertain edge (2,4) with probability p=0.6 and random weight Uniform(0.8,2.0).
    - Compute shortest path length from 1 to 5.
    """)
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    known_edges = [(1,2), (2,3), (3,4), (4,5)]
    G.add_edges_from(known_edges)
    
    p_edge = 0.6
    num_samples = 300
    missing_samples = np.random.uniform(0.8,2.0, num_samples)
    sp_lengths = []
    for val in missing_samples:
        G_temp = G.copy()
        # Add edge (2,4) with prob p_edge
        if np.random.rand() < p_edge:
            G_temp.add_edge(2,4, weight=val)
        # Known edges have weight=1
        for (u,v) in G_temp.edges():
            if "weight" not in G_temp[u][v]:
                G_temp[u][v]["weight"] = 1
        # Shortest path from 1 to 5
        sp = nx.shortest_path_length(G_temp, source=1, target=5, weight="weight")
        sp_lengths.append(sp)
    
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,4))
    sns.histplot(sp_lengths, bins=25, kde=True, color=cs["node_color"])
    plt.axvline(np.mean(sp_lengths), color=cs["kde_color"], linestyle="dashed", label="Mean")
    plt.title("Shortest Path (1 → 5) with Uncertain Edge (2,4)")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_monte_carlo_financial(bw_mode):
    """
    Financial Risk with missing transaction (F2,F4) ~ lognormal(2.0,0.5).
    """
    st.write(r"""
    **Financial Network with Missing Transaction**

    - Nodes: institutions (F1,F2,F3,F4).
    - Known edges: (F1,F2)=10, (F2,F3)=15, (F3,F4)=20, (F4,F1)=8
    - Missing edge: (F2,F4), lognormal(2.0, 0.5).
    - Compute total exposure across samples.
    """)
    G = nx.Graph()
    edges_fin = [('F1','F2',10), ('F2','F3',15), ('F3','F4',20), ('F4','F1',8)]
    for u,v,w in edges_fin:
        G.add_edge(u,v, weight=w)

    num_samples = 300
    mu_ln, sigma_ln = 2.0, 0.5
    samples = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=num_samples)
    exposures = []
    for val in samples:
        G_temp = G.copy()
        G_temp.add_edge('F2','F4', weight=val)
        tot = sum(data for _,data in nx.get_edge_attributes(G_temp, "weight").items())
        exposures.append(tot)
    
    cs = get_color_scheme(bw_mode)
    plt.figure(figsize=(6,4))
    sns.histplot(exposures, bins=25, kde=True, color=cs["node_color"])
    plt.axvline(np.mean(exposures), color=cs["kde_color"], linestyle="dashed", label="Mean")
    plt.title("Total Financial Exposure (Missing (F2,F4))")
    plt.xlabel("Exposure (million USD)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# --------------------- STREAMLIT APP ---------------------
st.title("Network Plot Generator with Monte Carlo Integration")

section = st.sidebar.radio("Sections", ["Random Network Models", "Monte Carlo Examples"])
bw_mode = st.sidebar.checkbox("Black and White Mode", value=False)

if section == "Random Network Models":
    st.header("Random Network Models")
    choice = st.selectbox("Choose Model", ["Simple Graph", "Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"])
    if choice == "Simple Graph":
        plot_simple_graph(bw_mode)
    elif choice == "Erdős-Rényi":
        st.write("Erdős-Rényi random graph G(n,p). Adjust n and p below.")
        n_er = st.slider("Number of Nodes (n)", 5, 100, 25)
        p_er = st.slider("Edge Probability (p)", 0.0, 1.0, 0.2, step=0.05)
        plot_erdos_renyi(n_er, p_er, bw_mode)
    elif choice == "Watts-Strogatz":
        st.write("Watts-Strogatz small-world network. Adjust n, k, p.")
        n_ws = st.slider("Number of Nodes (n)", 5, 100, 25)
        k_ws = st.slider("Nearest Neighbors (k)", 2, 10, 4, step=2)
        p_ws = st.slider("Rewiring Probability (p)", 0.0, 1.0, 0.2, step=0.05)
        plot_watts_strogatz(n_ws, k_ws, p_ws, bw_mode)
    elif choice == "Barabási-Albert":
        st.write("Barabási-Albert scale-free network. Adjust n, m.")
        n_ba = st.slider("Number of Nodes (n)", 5, 200, 25)
        m_ba = st.slider("Edges to attach (m)", 1, 10, 2)
        plot_barabasi_albert(n_ba, m_ba, bw_mode)

elif section == "Monte Carlo Examples":
    st.header("Monte Carlo Integration Examples")
    mc_choice = st.selectbox("Choose Example", ["Business Partnership", "Supply Chain", "Financial Risk"])
    if mc_choice == "Business Partnership":
        plot_monte_carlo_business_partnership(bw_mode)
    elif mc_choice == "Supply Chain":
        plot_monte_carlo_supply_chain(bw_mode)
    elif mc_choice == "Financial Risk":
        plot_monte_carlo_financial(bw_mode)

