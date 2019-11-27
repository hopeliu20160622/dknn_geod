import yaml
import os

import tensorflow as tf
from cleverhans.dataset import MNIST, CIFAR10
from dataloader import SVHN
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear, Softmax
from cleverhans import serial

from pathlib import Path


def dataset_loader(dataset_name, nb_train, nb_test):
    datasets_parser = {'MNIST':MNIST,
                       'CIFAR10':CIFAR10,
                       'SVHN':SVHN}
    data_loader = datasets_parser[dataset_name]
    dataset = data_loader(train_start=0, train_end=nb_train, 
                          test_start=0, test_end=nb_test)
    return dataset


class ModelConfig(object):
  def __init__(self, config_file, root_dir, copy=1):
      with open(config_file, 'r') as stream:
          config = yaml.safe_load(stream)
      
      self.dataset_name = config['dataset_name']
      self.max_epochs = config['train_parameters']['max_epochs']
      self.learning_rate = config['train_parameters']['learning_rate']
      self.lr_scheduler_step_size = config['train_parameters']['lr_scheduler_step_size']
      self.batch_size = config['train_parameters']['nb_random_seeds']
      self.nb_random_seeds = config['train_parameters']['nb_random_seeds']
      self.weight_decay = config['train_parameters']['weight_decay']
      self.nb_train = config['train_parameters']['nb_train']
      self.nb_test = config['train_parameters']['nb_test']
      self.gpu_memory_fraction = config['train_parameters']['gpu_memory_fraction']
      self.nb_proto_neighbors = config['nn_parameters']['nb_proto_neighbors']
      self.nb_neighbors = config['nn_parameters']['nb_neighbors']
      self.nb_cali = config['nn_parameters']['nb_cali']
      self.backend = config['nn_parameters']['backend']
      self.hash_hypar = config['nn_parameters']['hash_hypar']
      
      self.root_dir = root_dir
      self.copy = copy

      if self.dataset_name=='MNIST':
        self.img_rows, self.img_cols, self.nchannels = 28, 28, 1
        self.nb_classes = 10
      elif self.dataset_name=='CIFAR10':
        self.img_rows, self.img_cols, self.nchannels = 32, 32, 3
        self.nb_classes = 10
      elif self.dataset_name=='SVHN':
        self.img_rows, self.img_cols, self.nchannels = 32, 32, 3
        self.nb_classes = 10
  
  def get_model_dir_name(self, root_dir=None):
    if not root_dir:
      assert self.root_dir
      root_dir = self.root_dir

    data_dir = self.dataset_name
    model_parent_dir = os.path.join(root_dir, data_dir.upper())
    model_path = ['nb_train_{}'.format(self.nb_train),
                  'lr_{}'.format(self.learning_rate),
                  'bs_{}'.format(self.batch_size),
                  str(self.copy)]
    model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    return model_dir

  def get_model(self):
    """The model for the picklable models tutorial.
    """
    if self.dataset_name == 'MNIST':
        nb_filters=64
        nb_classes=self.nb_classes
        input_shape=(None, 28, 28, 1)
        layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
                ReLU(),
                Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
                ReLU(),
                Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
                ReLU(),
                Flatten(),
                Linear(nb_classes),
                Softmax()]
    elif self.dataset_name == 'CIFAR10':
        nb_filters=64
        nb_classes=self.nb_classes
        input_shape=(None, 32, 32, 3)
        layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
                ReLU(),
                Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
                ReLU(),
                Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
                ReLU(),
                Flatten(),
                Linear(nb_classes),
                Softmax()]
    elif self.dataset_name == 'SVHN':
        nb_filters=64
        nb_classes=self.nb_classes
        input_shape=(None, 32, 32, 3)
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
  
  def load_model(self, model_dir):
    data_filepath = os.path.join(model_dir, "model.joblib")
    path = Path(data_filepath)

    if path.is_file():
      print('Loading model from:\n {}'.format(data_filepath)+'\n')
      model = serial.load(data_filepath)
    
    else:
      print('Model path {} does not exist'.format(data_filepath))
      
    return model

  def get_tensorflow_session(self):
    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return sess
