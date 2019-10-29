import numpy as np
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state

import pyximport
pyximport.install(setup_args={"script_args":["--compiler=unix"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

from shortest_path import shortest_path

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
# GEODESIC LANDMARKS
##########################################################################

def compute_landmarks(features, n_landmarks='auto', random_state=None):
        """Computes the landmarks in the training data that will be used
        when fitting the data set.
        Returns
        -------
        landmarks_: array-like, shape (landmarks,)
            The array of landmarks to use, or None, if all samples should
            be used.
        """
        n_samples = features.shape[0]
        n_components = features.shape[1]
        landmarks_= None

        if n_landmarks == 'auto':
            n_landmarks = min(n_components + 10, n_samples)

        random_state = check_random_state(random_state)

        landmarks_ = np.arange(n_samples)
        random_state.shuffle(landmarks_)
        landmarks_ = landmarks_[:n_landmarks]

        return landmarks_

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

def hard_landmarks_geodesics_euclidean_kernel(features, n_neighbors, n_landmarks='auto'):
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='auto',
                            metric='euclidean',
                            n_jobs=2)
    nbrs_.fit(features)
    kng = kneighbors_graph(X=nbrs_, n_neighbors=n_neighbors,
                           metric='euclidean',
                           mode='distance', n_jobs=2)
    
    landmarks = compute_landmarks(features, n_landmarks)
    dist_matrix_ = shortest_path(kng,
                                 method='D',
                                 directed=False,
                                 indices=landmarks).T

    kernel = approx_geodesic(features, dist_matrix_, landmarks, nbrs_)
    return kernel

def approx_geodesic(X, dist_matrix_, landmarks_, nbrs_):
    """Transform X.
    This is implemented by linking the points X into the graph of geodesic
    distances of the training data. First the `n_neighbors` nearest
    neighbors of X are found in the training data, and from these the
    shortest geodesic distances from each point in X to each landmark in
    the training data are computed in order to construct the kernel.
    The embedding of X is the projection of this kernel onto the
    embedding vectors of the training set.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """
    
    #X = check_array(X)
    distances, indices = nbrs_.kneighbors(X, return_distance=True)

    # (or to the landmarks, when executing L-Isomap) via the nearest
    # neighbors of X.
    # This can be done as a single array operation, but it potentially
    # takes a lot of memory.  To avoid that, use a loop:
    columns = landmarks_.shape[0]

    G_X = np.zeros((X.shape[0], columns))
    for i in range(X.shape[0]):
        G_X[i] = np.min((dist_matrix_[indices[i]] +
                         distances[i][:, None]), axis=0)

    G_X **= 2
    G_X *= -0.5

    return G_X