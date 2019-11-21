import numpy as np
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
#from fast_geodesic_knn import fast_geodesic_knn
from scipy import sparse
from fast_gknn import fast_gknn

import time
import hnswlib

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
    kernel = compute_geodesics_kernel(kernel)
    return kernel

def soft_geodesics_euclidean_tv_kernel(features, temperature):
    kernel = euclidean_tv_kernel(features)
    kernel = normalize_soften_kernel(kernel, temp=temperature)
    kernel = compute_geodesics_kernel(kernel)
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
    max_distance = np.max(kernel)+1
    kernel[kernel == 0]=max_distance
    return kernel

def hard_geodesics_euclidean_kernel_regular(features, n_neighbors):
    # Regular
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
    max_distance = np.max(kernel)+1
    kernel[kernel == 0]=max_distance
    return kernel

def hard_geodesics_euclidean_kernel_fast(features, n_neighbors, k = 5):
    # Regular
    #nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
    #                         algorithm='auto',
    #                         metric='euclidean',
    #                         n_jobs=2)
    #nbrs_.fit(features)
    #kng = kneighbors_graph(X=nbrs_, n_neighbors=n_neighbors,
    #                       metric='euclidean',
    #                       mode='distance', n_jobs=2)

    # Approximate Neighbors
    num_elements, dim = features.shape
    data_idx = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    p.add_items(features, data_idx)
    # The higher the ef, the slower, but better accuracy
    p.set_ef(n_neighbors + 25) # Needs to be higher than the number of neighbors

    # We add one to n_neighbors since we exclude the identity as a neighbor
    neighbors_idx, distances = p.knn_query(features, k = n_neighbors + 1)
    distances = np.sqrt(distances) # knn_query returs squared euclidean norm

    # Convert the result in matrix form
    I = np.array(list(range(num_elements))*(n_neighbors))
    J = neighbors_idx[:,1:].flatten('F')
    V = distances[:,1:].flatten('F')

    kng = sparse.coo_matrix((V,(I,J)),shape=(num_elements,num_elements))

    kernel = fast_gknn(kng, directed=False, k=k)
    return kernel

def hard_geodesics_euclidean_kernel_approx(features, n_neighbors):
    # Approximate Neighbors
    num_elements, dim = features.shape
    data_labels = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    p.add_items(features, data_labels)
    # The higher the ef, the slower, but better accuracy
    p.set_ef(n_neighbors + 25) # Needs to be higher than the number of neighbors

    # We add one to n_neighbors since we exclude the identity as a neighbor
    labels, distances = p.knn_query(features, k = n_neighbors + 1)
    distances = np.sqrt(distances) # knn_query returs squared euclidean norm

    # Convert the result in matrix form
    kng = np.zeros((num_elements, num_elements))
    for i, label in enumerate(labels):
        for index, neighbor in enumerate(label[1:]):
            kng[i][neighbor] = distances[i][index+1]
    kng[kng == 0] = np.Infinity
    # No change here on
    dist_matrix_ = graph_shortest_path(kng,
                                       method='FW',
                                       directed=False)
    kernel = (0.5)*dist_matrix_**2
    max_distance = np.max(kernel)+1
    kernel[kernel == 0]=max_distance
    return kernel

def hard_geodesics_euclidean_kernel_approx_sparse(features, n_neighbors):
    # Approximate Neighbors
    num_elements, dim = features.shape
    data_idx = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    p.add_items(features, data_idx)
    # The higher the ef, the slower, but better accuracy
    p.set_ef(n_neighbors + 25) # Needs to be higher than the number of neighbors

    # We add one to n_neighbors since we exclude the identity as a neighbor
    neighbors_idx, distances = p.knn_query(features, k = n_neighbors + 1)
    distances = np.sqrt(distances) # knn_query returs squared euclidean norm

    # Convert the result in matrix form
    I = np.array(list(range(num_elements))*(n_neighbors))
    J = neighbors_idx[:,1:].flatten('F')
    V = distances[:,1:].flatten('F')

    kng = sparse.coo_matrix((V,(I,J)),shape=(num_elements,num_elements))

    # No change here on
    dist_matrix_ = graph_shortest_path(kng,
                                       method='FW',
                                       directed=False)
    kernel = (0.5)*dist_matrix_
    max_distance = np.max(kernel)+1
    kernel[kernel == 0]=max_distance
    return kernel


def hard_geodesics_euclidean_kernel(features, n_neighbors, k):
    #TODO: add the new kernel to DkNNModel class instead of switching here

    #return hard_geodesics_euclidean_kernel_regular(features, n_neighbors)
    #return hard_geodesics_euclidean_kernel_approx(features, n_neighbors)
    #return hard_geodesics_euclidean_kernel_approx_sparse(features, n_neighbors)
    return hard_geodesics_euclidean_kernel_fast(features, n_neighbors, k)