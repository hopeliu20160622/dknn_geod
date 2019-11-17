import numpy as np
import scipy.sparse

import heapdict

heap_pop_count = 0

def check_sparse_edge_weights_matrix(W):
    assert type(W) == scipy.sparse.csr.csr_matrix
    (n, n_other) = W.shape
    assert n == n_other
    assert (W.data >= 0).all()
    #assert (W.transpose() != W).nnz == 0
    return n

def fast_geodesic_knn(W, labeled_mask, k):
    '''
    Input:
        W: n by n scipy.sparse.csr_matrix
            Edge *symmetric* weight matrix. We use the scipy.sparse.csgraph convention that non-edges are denoted by non-entries.

        labeled_mask: boolean array of length n indicating which vertices are labeled

    Output:
        knn: knn[i] is a list of up to k pairs of (dist, seed)
    '''
    global heap_pop_count

    n = check_sparse_edge_weights_matrix(W)
    assert labeled_mask.dtype == np.bool
    assert labeled_mask.shape == (n,)

    labeled_vertex_indices = labeled_mask.nonzero()[0]

    visited = set()
    knn = [[] for i in range(n)]
    heap = heapdict.heapdict()
    for s in labeled_vertex_indices:
        heap[(s, s)] = 0.0

    W_indptr = W.indptr
    W_indices = W.indices
    W_data = W.data
    while len(heap) > 0:
        ((seed, i), dist_seed_i) = heap.popitem()
        heap_pop_count += 1
        visited.add((seed, i))

        if len(knn[i]) < k:
            knn[i].append((dist_seed_i, seed))

            for pos in range(W_indptr[i], W_indptr[i+1]):
                j = W_indices[pos]
                if (seed, j) not in visited:
                    alt_dist = dist_seed_i + W_data[pos]
                    if (seed, j) not in heap or alt_dist < heap[(seed, j)]:
                        heap[(seed, j)] = alt_dist

    return knn

if __name__ == '__main__':
    import numpy as np
    import scipy.sparse

    import fast_geodesic_knn

    def build_sparse_undirected_nonnegative_edge_matrix(n):
        W = np.random.random((n,n))
        W = W + W.transpose()
        W[W < 0.5] = np.inf
        return scipy.sparse.csr_matrix(W)

    def test_geodesic_knn():
        N = 10
        p = 0.2 
        k = 2
        
        W = build_sparse_undirected_nonnegative_edge_matrix(N)
        labeled_mask = np.random.random(N) < p
        print('labeled vertices:')
        print(labeled_mask.nonzero()[0])

        result0 = fast_geodesic_knn.fast_geodesic_knn(W, labeled_mask, k)

        for i in range(len(result0)):
            print('result0[{0}]:'.format(i))
            print(result0[i])
    test_geodesic_knn()
