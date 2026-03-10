import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp

def load_city_graph(file_path):
    """
    Reads a City Graph .pkl file and returns the Adjacency Matrix (A)
    and Node Feature Matrix (X) required for the competition.
    
    Args:
        file_path (str): Path to the .pkl file (e.g., 'data/train/Beijing.pkl')
        
    Returns:
        A (scipy.sparse.coo_matrix): The Adjacency Matrix (N x N)
        X (numpy.ndarray): The Node Feature Matrix (N x 2) representing (x, y) coordinates.
    """
    # 1. Load the graph object
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
        
    # 2. Get consistent node ordering
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # 3. Build Adjacency Matrix (A)
    # We use a sparse matrix because city graphs are sparse (efficient!)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format='coo')
    
    # 4. Build Node Feature Matrix (X)
    # Extract the 'x' (longitude) and 'y' (latitude) attributes you saved
    xs = np.array([G.nodes[n].get('x', 0) for n in nodes])
    ys = np.array([G.nodes[n].get('y', 0) for n in nodes])
    
    # Center the coordinates (Normalization)
    # This helps GNNs learn better
    if len(xs) > 0:
        xs = xs - xs.mean()
        ys = ys - ys.mean()
        
    # Stack into [N, 2] matrix
    X = np.stack([xs, ys], axis=1)
    
    return A, X, G.graph.get('target', None)