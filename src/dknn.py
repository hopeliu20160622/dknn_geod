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

if 'DISPLAY' not in os.environ:
  matplotlib.use('Agg')

FLAGS = tf.flags.FLAGS

def make_basic_picklable_cnn(nb_filters=64, nb_classes=10,
                             input_shape=(None, 28, 28, 1)):
  """The model for the picklable models tutorial.
  """
  layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Linear(nb_classes),
            Softmax()]
  model = MLP(layers, input_shape)
  return model

class DkNNModel(Model):
  def __init__(self, neighbors, layers, get_activations, train_data, train_labels,
               nb_classes, scope=None, nb_tables=200):
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

def get_tensorflow_session():
    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction=FLAGS.tensorflow_gpu_memory_fraction
    sess = tf.Session(
        config=tf.ConfigProto(
            gpu_options=gpu_options
        )
    )

    return sess

def compute_geodesic_matrices():
  mnist = MNIST(train_start=0, train_end=FLAGS.nb_train, test_start=0, test_end=1000)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters.
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  with get_tensorflow_session() as sess:
    with tf.variable_scope('dknn'):
      tf.set_random_seed(FLAGS.seed)
      np.random.seed(int(FLAGS.seed))

      # Define input TF placeholder.
      x = tf.placeholder(tf.float32, shape=(
        None, img_rows, img_cols, nchannels))
      y = tf.placeholder(tf.float32, shape=(None, nb_classes))

      # Define a model.
      model = make_basic_picklable_cnn()
      preds = model.get_logits(x)
      loss = CrossEntropy(model, smoothing=0.)

      # Define the test set accuracy evaluation.
      def evaluate():
          acc = model_eval(sess, x, y, preds, x_test, y_test,
                            args={'batch_size': FLAGS.batch_size})
          print('Test accuracy on test examples: %0.4f' % acc)

      # Train the model
      train_params = {'nb_epochs': FLAGS.nb_epochs,
                    'batch_size': FLAGS.batch_size, 'learning_rate': FLAGS.lr}

      model_filepath = "../data/model.joblib"
      path = Path(model_filepath)

      if path.is_file():
          model = serial.load(model_filepath)
      else:
          train(sess, loss, x_train, y_train, evaluate=evaluate,
            args=train_params, var_list=model.get_params())
          serial.save(model_filepath, model)

        # Define callable that returns a dictionary of all activations for a dataset
      def get_activations(data):
          data_activations = {}
          for layer in layers:
              layer_sym = tf.layers.flatten(model.get_layer(x, layer))
              data_activations[layer] = batch_eval(sess, [x], [layer_sym], [data],
                                                  args={'batch_size': FLAGS.batch_size})[0]
          return data_activations

      # Use a holdout of the test set to simulate calibration data for the DkNN.
      train_data = x_train
      train_labels = np.argmax(y_train, axis=1)
      cali_data = x_test[:FLAGS.nb_cali]
      y_cali = y_test[:FLAGS.nb_cali]
      cali_labels = np.argmax(y_cali, axis=1)
      test_data = x_test[FLAGS.nb_cali:]
      y_test = y_test[FLAGS.nb_cali:]

      # Extract representations for the training and calibration data at each layer of interest to the DkNN.
      layers = ['ReLU1', 'ReLU3', 'ReLU5', 'logits']

      # Wrap the model into a DkNNModel
      dknn = DkNNModel(
      FLAGS.neighbors,
      layers,
      get_activations,
      train_data,
      train_labels,
      nb_classes,
      scope='dknn'
      )

  # Compute matrix for each layer
  geodesic_matrices = []
  for layer in layers:
    print(layer)
    activations = dknn.train_activations[layer]
    geodesic_matrix = hard_geodesics_euclidean_kernel(activations, FLAGS.proto_neighbors)
    geodesic_matrices.append(geodesic_matrix)

  matrix_path = '../results/geodesic_matrices_'+str(FLAGS.nb_train)+'_'+str(FLAGS.proto_neighbors) + '.pkl'
  with open(matrix_path, 'wb') as f:
    pickle.dump(geodesic_matrices, f)

  return True

def main(argv=None):
  assert compute_geodesic_matrices()


if __name__ == '__main__':
  tf.flags.DEFINE_integer(
    'number_bits',
    17,
    'number of hash bits used by LSH Index'
  )
  tf.flags.DEFINE_float(
    'tensorflow_gpu_memory_fraction',
    0.25,
    'amount of the GPU memory to allocate for a tensorflow Session'
  )
  tf.flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
  tf.flags.DEFINE_integer('batch_size', 500, 'Size of training batches')
  tf.flags.DEFINE_float('lr', 0.001, 'Learning rate for training')
  tf.flags.DEFINE_float('seed', 123, 'Random seed')
  tf.flags.DEFINE_integer(
      'nb_cali', 750, 'Number of calibration points for the DkNN')
  tf.flags.DEFINE_integer(
      'neighbors', 75, 'Number of neighbors per layer for the DkNN')
  tf.flags.DEFINE_integer('proto_neighbors', 5,
      'Number of neighbors for the sparse adjacency for the geodesic kernel')
  tf.flags.DEFINE_integer('nb_train', 1000,
      'Number of train observations')
  tf.app.run()
