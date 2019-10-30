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
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.model import Model
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear, Softmax
from cleverhans.train import train
from cleverhans.utils_tf import batch_eval, model_eval
from cleverhans import serial
from pathlib import Path
from utils_kernel import euclidean_kernel, hard_geodesics_euclidean_kernel

class NNGeod():
  def __init__(self, neighbors, proto_neighbors):
    self.nb_neighbors = neighbors
    self.nb_proto_neighbors = proto_neighbors
  
  def closest_neighbor(self, x):
    deltas = self.X - x
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)
  
  def add(self, X):
    self.geodesic_kernel = hard_geodesics_euclidean_kernel(X, self.nb_proto_neighbors)
    self.train_neighbor_index = np.argpartition(self.geodesic_kernel, 
                                                   kth=self.nb_neighbors-1, axis=1)[:,:self.nb_neighbors-1]
    self.X = X
    return self

  def find_knns(self, x, output):
    closest_neighbor_index = self.closest_neighbor(x)
    neighbor_index = self.train_neighbor_index[closest_neighbor_index, :]
    neighbor_index = np.append(closest_neighbor_index, neighbor_index)

    missing_indices = [False] * self.nb_neighbors

    d1 = neighbor_index.reshape(-1)

    output.reshape(-1)[
      np.logical_not(missing_indices.flatten())
    ] = d1[
      np.logical_not(missing_indices.flatten())
    ]

    return missing_indices

class DkNNModel(Model):
  def __init__(self, neighbors, proto_neighbors, layers, get_activations, train_data, train_labels,
               nb_classes, method, scope=None, nb_tables=200):
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
    self.neighbors = neighbors
    self.proto_neighbors = proto_neighbors
    self.method = method
    self.nb_tables = nb_tables
    self.layers = layers
    self.get_activations = get_activations
    self.nb_cali = -1
    self.calibrated = False

    # Compute training data activations
    self.nb_train = train_labels.shape[0]
    assert self.nb_train == train_data.shape[0]
    self.train_activations = get_activations(train_data)
    self.train_labels = train_labels

    # Build locality-sensitive hashing tables for training representations
    self.train_activations_querier = copy.copy(self.train_activations)
    self.init_nn_querier()

  def init_nn_querier(self):
    """
    Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
    """
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

      if self.method == 'euclidean':
        print('Constructing the NearestNeighbor table')
        # self.query_objects[layer] = NearestNeighbor(
        #  backend=2,
        #  dimension=self.train_activations_querier[layer].shape[1],
        #  number_bits=self.number_bits,
        #  neighbors=self.neighbors,
        #  nb_tables=self.nb_tables
        # )
        # self.query_objects[layer].add(self.train_activations_querier[layer])

      elif self.method == 'geodesic':
        print('Constructing the NearestNeighborGeodesic table')
        self.query_objects[layer] = NNGeod(self.neighbors, self.proto_neighbors)
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

    print("Starting calibration of DkNN.")
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
    print("DkNN calibration complete.")
