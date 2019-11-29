"""
This code reproduces the MNIST results from the paper
Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning
https://arxiv.org/abs/1803.04765
The LSH backend used in the paper is FALCONN. This script also demonstrates
how to use an alternative backend called FAISS.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
from bisect import bisect_left
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from six.moves import xrange
import enum
import tensorflow as tf
import time
import hnswlib
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.model import Model
from cleverhans.train import train
from cleverhans.evaluation import batch_eval
from cleverhans import serial
from pathlib import Path
from utils_kernel import euclidean_kernel, hard_geodesics_euclidean_kernel
from scipy.spatial import distance
from fast_gknn import fast_gknn
from scipy import sparse

###################################
# TENSORFLOW UTILS
###################################

FLAGS = tf.flags.FLAGS

###################################
# PAPERNOT NEAREST NEIGHBOR CLASS
###################################

class NearestNeighbor:
  #class BACKEND(enum.Enum):
  #  FALCONN = 1
  #  FAISS = 2

  def __init__(
    self,
    backend,
    dimension,
    neighbors,
    number_bits,
    nb_tables=None,
    hash_table_path=None
  ):
    #assert backend in NearestNeighbor.BACKEND

    self._NEIGHBORS = neighbors
    self._BACKEND = backend

    if self._BACKEND == 'FALCONN':
      self._init_falconn(
        dimension,
        number_bits,
        nb_tables
      )
    elif self._BACKEND == 'FAISS':
      self._init_faiss(
        dimension,
      )
    else:
      raise NotImplementedError

  def _init_falconn(
    self,
    dimension,
    number_bits,
    nb_tables,
  ):
    import falconn

    if self._NEIGHBORS < 512:
      nb_tables = 200
    else:
      nb_tables = 600

    assert nb_tables >= self._NEIGHBORS

    # LSH parameters
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = dimension
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = nb_tables
    params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

    # we build number_bits-bit hashes so that each table has
    # 2^number_bits bins; a rule of thumb is to have the number
    # of bins be the same order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(number_bits, params_cp)
    self._falconn_table = falconn.LSHIndex(params_cp)
    self._falconn_query_object = None
    self._FALCONN_NB_TABLES = nb_tables

  def _init_faiss(
    self,
    dimension,
  ):
    import faiss

    res = faiss.StandardGpuResources()

    self._faiss_index = faiss.GpuIndexFlatL2(
      res,
      dimension,
    )

  def _find_knns_falconn(self, x, output, nb_neighbors):
    # Late falconn query_object construction
    # Since I suppose there might be an error
    # if table.setup() will be called after
    if self._falconn_query_object is None:
      self._falconn_query_object = self._falconn_table.construct_query_object()
      self._falconn_query_object.set_num_probes(
        self._FALCONN_NB_TABLES
      )

    missing_indices = np.zeros(output.shape, dtype=np.bool)

    for i in range(x.shape[0]):
      query_res = self._falconn_query_object.find_k_nearest_neighbors(
        x[i],
        nb_neighbors
      )
      try:
        output[i, :] = query_res
      except:  # pylint: disable-msg=W0702
        # mark missing indices
        missing_indices[i, len(query_res):] = True

        output[i, :len(query_res)] = query_res

    return missing_indices

  def _find_knns_faiss(self, x, output, nb_neighbors):
    neighbor_distance, neighbor_index = self._faiss_index.search(
      x,
      nb_neighbors
    )

    missing_indices = neighbor_distance == -1

    d1 = neighbor_index.reshape(-1)

    output.reshape(-1)[
      np.logical_not(missing_indices.flatten())
    ] = d1[
      np.logical_not(missing_indices.flatten())
    ]

    return missing_indices

  def add(self, x):
    if self._BACKEND == 'FALCONN':
      self._falconn_table.setup(x)
    elif self._BACKEND == 'FAISS':
      self._faiss_index.add(x)
    else:
      raise NotImplementedError

  def find_knns(self, x, output, nb_neighbors=None):
    if nb_neighbors is None:
      nb_neighbors=self._NEIGHBORS
    if self._BACKEND == 'FALCONN':
      return self._find_knns_falconn(x, output, nb_neighbors)
    elif self._BACKEND == 'FAISS':
      return self._find_knns_faiss(x, output, nb_neighbors)
    else:
      raise NotImplementedError

  def update_neighbors(self, neighbors):
        self._NEIGHBORS = neighbors

###################################
# HASHING  NEAREST NEIGHBOR CLASS
###################################

class NNHash():
  def __init__(self, neighbors, dimension, hash_hypar, hash_table_path):
    self.nb_neighbors = neighbors
    self.dimension = dimension
    self.hash_hypar = hash_hypar
    self.hash_table_path = hash_table_path

  def add(self, X):
    nb_obs, dim = X.shape
    data_idx = np.arange(nb_obs)

    self.hash_table = hnswlib.Index(space = 'l2', dim = dim)
    if os.path.exists(self.hash_table_path):
      self.hash_table.load_index(self.hash_table_path, max_elements = nb_obs)
      self.hash_table.set_ef(self.hash_hypar)
    else:
      self.hash_table.init_index(max_elements = nb_obs, ef_construction = self.hash_hypar, M = 48)
      self.hash_table.set_ef(self.hash_hypar)
      self.hash_table.add_items(X, data_idx)
      self.hash_table.save_index(self.hash_table_path)

  def find_knns(self, x, output):
    neighbors_index, _ = self.hash_table.knn_query(x, k=self.nb_neighbors)

    for i in range(x.shape[0]):
      output[i, :] = neighbors_index[i]
    
    missing_indices = np.zeros(output.shape, dtype=np.bool)
    return missing_indices

  def update_neighbors(self, neighbors):
    self.nb_neighbors = neighbors

###################################
# GEODEISC NEAREST NEIGHBOR CLASS
###################################

class NNGeod():
  def __init__(self, backend, number_bits, neighbors, dimension, nb_tables, proto_neighbors, neighbors_table_path, hash_table_path):
    self.nb_neighbors = neighbors
    self.dimension = dimension
    self.hash_hypar = nb_tables
    self.nb_proto_neighbors = proto_neighbors
    self.neighbors_table_path = neighbors_table_path
    self.hash_table_path = hash_table_path

  def compute_geodesic_neighbors(self, features):
    # Approximate Neighbors
    nb_obs, dim = features.shape

    self.NN = NNHash(self.nb_proto_neighbors+1, self.dimension, self.hash_hypar, self.hash_table_path)
    self.NN.add(features)

    if os.path.exists(self.neighbors_table_path):
      gknn = np.load(self.neighbors_table_path)
    else:
      neighbors_idx, distances = self.NN.hash_table.knn_query(features, k=self.nb_proto_neighbors+1)
      distances = np.sqrt(distances) # knn_query returs squared euclidean norm

      # Convert the result in matrix form
      I = np.array(list(range(nb_obs))*(self.nb_proto_neighbors))
      J = neighbors_idx[:,1:].flatten('F')
      V = distances[:,1:].flatten('F')

      kng = sparse.coo_matrix((V,(I,J)),shape=(nb_obs,nb_obs))
      gknn = fast_gknn(kng, directed=False, k=self.nb_neighbors)

      np.save(self.neighbors_table_path, gknn)
    return gknn
  
  def add(self, X):
    self.gknn = self.compute_geodesic_neighbors(features=X)
    self.train_neighbor_index = self.gknn[:,:,1][:,1:].astype(int)[:,:self.nb_neighbors-1]
    self.X = X

    return self

  def find_knns(self, x, output):
    closest_neighbor_index, _ = self.NN.hash_table.knn_query(x, k=1)

    for i in range(x.shape[0]):
      neighbor_index = self.train_neighbor_index[closest_neighbor_index[i], :]
      neighbor_index = np.append(closest_neighbor_index[i], neighbor_index)
      output[i, :] = neighbor_index
    
    missing_indices = np.zeros(output.shape, dtype=np.bool)

    return missing_indices

  def update_neighbors(self, neighbors):
    self.nb_neighbors = neighbors
    self.train_neighbor_index = self.gknn[:,:,1][:,1:].astype(int)[:,:self.nb_neighbors-1]


###################################
# GREAT MODEL
###################################

class DkNNModel(Model):
  def __init__(self, sess, model, backend, neighbors, proto_neighbors, layers, 
               train_data, train_labels, img_rows, img_cols, nchannels, nb_classes, 
               method, neighbors_table_path=None, hash_table_path=None, scope=None, number_bits=17, nb_tables=600):
    """
    Implements the DkNN algorithm. See https://arxiv.org/abs/1803.04765 for more details.
    :param neighbors: number of neighbors to find per layer.
    :param layers: a list of layer names to include in the DkNN.
    :param get_activations: a callable that takes a np array and a layer name and returns its activations on the data.
    :param train_data: a np array of training data.
    :param train_labels: a np vector of training labels.
    :param nb_classes: the number of classes in the task.
    :param scope: a TF scope that was used to create the underlying model.
    :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
    """
    super(DkNNModel, self).__init__(nb_classes=nb_classes, scope=scope)
    self.sess = sess
    self.model = model
    self.neighbors = neighbors
    self.proto_neighbors = proto_neighbors
    self.method = method
    self.nb_tables = nb_tables
    self.layers = layers
    self.nb_cali = -1
    self.calibrated = False
    self.number_bits = number_bits
    self.backend = backend

    # Compute training data activations
    self.nb_train = train_labels.shape[0]
    assert self.nb_train == train_data.shape[0]

    # Input data dimensions
    self.img_rows, self.img_cols, self.nchannels = img_rows, img_cols, nchannels
    self.nb_classes = nb_classes
    
    self.train_labels = train_labels
    self.neighbors_table_path = neighbors_table_path
    self.hash_table_path = hash_table_path

    # Build graph
    self.build_graph()
    self.train_activations = self.get_activations(train_data)
  
    # Define callable that returns a dictionary of all activations for a dataset
  def build_graph(self):
      self.x_ph = tf.placeholder(tf.float32, shape=(None, self.img_rows, self.img_cols, self.nchannels))
      self.y_ph = tf.placeholder(tf.float32, shape=(None, self.nb_classes))
      
      fprop = self.model.fprop(self.x_ph)
      self.layer_sym_ph = {layer: tf.layers.flatten(fprop[layer]) for layer in self.layers}
  
  # Define callable that returns a dictionary of all activations for a dataset
  def get_activations(self, data, batch_size=10):
      data_activations = {}
      for layer in self.layers:
          data_activations[layer] = batch_eval(sess=self.sess, 
                                               tf_inputs=[self.x_ph],
                                               tf_outputs=[self.layer_sym_ph[layer]],
                                               numpy_inputs=[data], batch_size=batch_size)[0]
      return data_activations

  def update_neighbors(self, neighbors):
      self.neighbors = neighbors
      for layer in self.layers:
        self.query_objects[layer].update_neighbors(self.neighbors)

  def fit(self):
    """
    Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
    """
    # Build locality-sensitive hashing tables for training representations
    self.train_activations_querier = copy.copy(self.train_activations)

    self.query_objects = {
    }  # contains the object that can be queried to find nearest neighbors at each layer.
    # mean of training data representation per layer (that needs to be substracted before
    # NearestNeighbor).
    self.centers = {}
    for layer in self.layers:
      # Normalize all the lenghts, since we care about the cosine similarity.
      self.train_activations_querier[layer] /= np.linalg.norm(
          self.train_activations_querier[layer], axis=1).reshape(-1, 1)

      # Center the dataset and the queries: this improves the performance of LSH quite a bit.
      center = np.mean(self.train_activations_querier[layer], axis=0)
      self.train_activations_querier[layer] -= center
      self.centers[layer] = center

      layer_hash_path = os.path.join(self.neighbors_table_path, 'hash_{}.npy'.format(layer))
      if self.method == 'euclidean':
        print('Constructing the NearestNeighbor table layer {}'.format(layer))
        self.query_objects[layer] = NearestNeighbor(neighbors=self.neighbors,
                                                    backend = self.backend,
                                                    number_bits = self.number_bits,
                                                    dimension=self.train_activations_querier[layer].shape[1],
                                                    nb_tables=self.nb_tables,
                                                    hash_table_path = layer_hash_path)
        self.query_objects[layer].add(self.train_activations_querier[layer])

      elif self.method == 'geodesic':
        print('Constructing the GeodesicNearestNeighbor table layer {}'.format(layer))
        layer_geodesics_path = os.path.join(self.neighbors_table_path, 'geodesics_{}.npy'.format(layer))

        self.query_objects[layer] = NNGeod(neighbors = self.neighbors,
                                           backend = self.backend,
                                           number_bits = self.number_bits,
                                           dimension=self.train_activations_querier[layer].shape[1],
                                           nb_tables=self.nb_tables,
                                           proto_neighbors = self.proto_neighbors,
                                           neighbors_table_path = layer_geodesics_path,
                                           hash_table_path = layer_hash_path)
        self.query_objects[layer].add(self.train_activations_querier[layer])

  def find_train_knns(self, data_activations):
    """
    Given a data_activation dictionary that contains a np array with activations for each layer,
    find the knns in the training data.
    """
    knns_ind = {}
    knns_labels = {}

    for layer in self.layers:
      # Pre-process representations of data to normalize and remove training data mean.
      data_activations_layer = copy.copy(data_activations[layer])
      nb_data = data_activations_layer.shape[0]
      data_activations_layer /= np.linalg.norm(
          data_activations_layer, axis=1).reshape(-1, 1)
      data_activations_layer -= self.centers[layer]

      # Use FALCONN to find indices of nearest neighbors in training data.
      knns_ind[layer] = np.zeros(
          (data_activations_layer.shape[0], self.neighbors), dtype=np.int32)
      knn_errors = 0

      knn_missing_indices = self.query_objects[layer].find_knns(
        data_activations_layer,
        knns_ind[layer],
      )

      knn_errors += knn_missing_indices.flatten().sum()

      # Find labels of neighbors found in the training data.
      knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)

      knns_labels[layer].reshape(-1)[
        np.logical_not(
          knn_missing_indices.flatten()
        )
      ] = self.train_labels[
        knns_ind[layer].reshape(-1)[
          np.logical_not(
            knn_missing_indices.flatten()
          )
        ]
      ]

    return knns_ind, knns_labels

  def nonconformity(self, knns_labels):
    """
    Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
    each candidate label for each data point: i.e. the number of knns whose label is
    different from the candidate label.
    """
    nb_data = knns_labels[self.layers[0]].shape[0]
    knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)
    for i in range(nb_data):
      # Compute number of nearest neighbors per class
      knns_in_class = np.zeros(
          (len(self.layers), self.nb_classes), dtype=np.int32)
      for layer_id, layer in enumerate(self.layers):
        knns_in_class[layer_id, :] = np.bincount(
            knns_labels[layer][i], minlength=self.nb_classes)

      # Compute number of knns in other class than class_id
      for class_id in range(self.nb_classes):
        knns_not_in_class[i, class_id] = np.sum(
            knns_in_class) - np.sum(knns_in_class[:, class_id])
    return knns_not_in_class

  def preds_conf_cred(self, knns_not_in_class):
    """
    Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
    the DkNN's prediction, confidence and credibility.
    """
    nb_data = knns_not_in_class.shape[0]
    preds_knn = np.zeros(nb_data, dtype=np.int32)
    confs = np.zeros((nb_data, self.nb_classes), dtype=np.float32)
    creds = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

    for i in range(nb_data):
      # p-value of test input for each class
      p_value = np.zeros(self.nb_classes, dtype=np.float32)

      for class_id in range(self.nb_classes):
        # p-value of (test point, candidate label)
        p_value[class_id] = (float(self.nb_cali) - bisect_left(
            self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

      preds_knn[i] = np.argmax(p_value)
      confs[i, preds_knn[i]] = 1. - p_value[np.argsort(p_value)[-2]]
      creds[i, preds_knn[i]] = p_value[preds_knn[i]]
    return preds_knn, confs, creds

  def fprop_np(self, data_np):
    """
    Performs a forward pass through the DkNN on an numpy array of data.
    """
    if not self.calibrated:
      raise ValueError(
          "DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring.")
    data_activations = self.get_activations(data_np)
    _, knns_labels = self.find_train_knns(data_activations)
    knns_not_in_class = self.nonconformity(knns_labels)
    _, _, creds = self.preds_conf_cred(knns_not_in_class)
    return creds

  def fprop(self, x):
    """
    Performs a forward pass through the DkNN on a TF tensor by wrapping
    the fprop_np method.
    """
    logits = tf.py_func(self.fprop_np, [x], tf.float32)
    return {self.O_LOGITS: logits}

  def calibrate(self, cali_data, cali_labels):
    """
    Runs the DkNN on holdout data to calibrate the credibility metric.
    :param cali_data: np array of calibration data.
    :param cali_labels: np vector of calibration labels.
    """
    self.nb_cali = cali_labels.shape[0]
    self.cali_activations = self.get_activations(cali_data)
    self.cali_labels = cali_labels

    print("Starting calibration.")
    cali_knns_ind, cali_knns_labels = self.find_train_knns(
        self.cali_activations)

    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_ind.values()])
    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_labels.values()])

    cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
    cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
    for i in range(self.nb_cali):
      cali_knns_not_in_l[i] = cali_knns_not_in_class[i, cali_labels[i]]
    cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
    self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
    self.nb_cali = self.cali_nonconformity.shape[0]
    self.calibrated = True
    print("Completed calibration.\n")

  def predict(self, new_x):
    #print("Predicting.")
    # Get activations
    new_activations = self.get_activations(new_x)
    # Get neighbors
    new_knns_ind, new_knns_labels = self.find_train_knns(new_activations)

    assert all([v.shape == (new_x.shape[0], self.neighbors)
               for v in new_knns_ind.values()])
    assert all([v.shape == (new_x.shape[0], self.neighbors)
               for v in new_knns_ind.values()])

    # Nonconformity
    new_knns_not_in_class = self.nonconformity(new_knns_labels)
    # Predictions
    preds, confs, creds = self.preds_conf_cred(new_knns_not_in_class)
    #print("Prediction complete.")
    return preds, confs, creds
