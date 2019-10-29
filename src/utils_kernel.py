import numpy as np
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

##########################################################################
# KERNEL TRANSFORMATIONS
##########################################################################

def normalize_soften_kernel(kernel, temp=1):
    exp_matrix = np.exp((1/temp) * kernel)
    norm = np.sum(exp_matrix, axis=1)
    soft_kernel_matrix = exp_matrix / norm[:, np.newaxis]
    return soft_kernel_matrix

def compute_geodesics_kernel(kernel):
    # weight = -log(P)
    weight_matrix = -np.log(kernel)
    # shortest path <=> likeliest path
    geodesics_kernel = -graph_shortest_path(weight_matrix, method='FW')
    return geodesics_kernel

##########################################################################
# KERNEL DEFINITIONS
##########################################################################

def cosine_kernel(features):
    kernel = cosine_similarity(features)
    return kernel

def euclidean_kernel(features):
    euclidean_distances_sq = euclidean_distances(features, squared=True)
    kernel = (0.5)*euclidean_distances_sq
    return kernel

def euclidean_tv_kernel(features):
    # compute kernel similarity induced by euclidean distance
    # normalize using total variation of the features
    euclidean_distances_sq = euclidean_distances(features, squared=True)
    #total_variation_orig = np.trace(np.cov(features.T)) #potential quadratic problem
    total_variation = np.sum(np.var(features, axis=0))
    # biasing total_variation because of curvature/correlations of features
    # if too many features one over normalizes
    kernel = -(1/total_variation) * euclidean_distances_sq
    return kernel

def soft_geodesics_cosine_kernel(features, temperature):
    kernel = cosine_similarity(features)
    kernel = normalize_soften_kernel(kernel, temp=temperature)
    kernel = compute_shortest_paths_matrix(normalized_kernel)
    return kernel

def soft_geodesics_euclidean_tv_kernel(features, temperature):
    kernel = euclidean_similarity_tv(features)
    kernel = normalize_soften_kernel(kernel, temp=temperature)
    kernel = compute_shortest_paths_matrix(kernel)
    return kernel

def hard_geodesics_euclidean_tv_kernel(features, n_neighbors):
    total_variation_sqrt = np.sqrt(np.trace(np.cov(features.T)))
    normalized_features = (1/total_variation_sqrt) * features
    
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='auto',
                            metric='euclidean',
                            n_jobs=2)
    nbrs_.fit(normalized_features)
    kng = kneighbors_graph(X=nbrs_, n_neighbors=n_neighbors,
                           metric='euclidean',
                           mode='distance', n_jobs=2)

    dist_matrix_ = graph_shortest_path(kng,
                                       method='FW',
                                       directed=False)
    kernel = (0.5)*dist_matrix_**2
    return kernel

def hard_geodesics_euclidean_kernel(features, n_neighbors):
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='auto',
                            metric='euclidean',
                            n_jobs=2)
    nbrs_.fit(features)
    kng = kneighbors_graph(X=nbrs_, n_neighbors=n_neighbors,
                           metric='euclidean',
                           mode='distance', n_jobs=2)

    dist_matrix_ = graph_shortest_path(kng,
                                       method='FW',
                                       directed=False)
    kernel = (0.5)*dist_matrix_**2
    return kernel