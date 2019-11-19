import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from cleverhans.train import train
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import batch_eval, model_eval
from cleverhans import serial

from utils_config import ModelConfig, dataset_loader
from dknn import DkNNModel


def train_model(mc):
    dataset = dataset_loader(dataset_name=mc.dataset_name,
                             nb_train=mc.nb_train,
                             nb_test=mc.nb_test)
    x_train, y_train = dataset.get_set('train')
    x_test, y_test = dataset.get_set('test')
    
    # Use Image Parameters.
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    
    with mc.get_tensorflow_session() as sess:
        with tf.variable_scope('dknn'):
            # Define input TF placeholder.
            x = tf.placeholder(tf.float32,
                               shape=(None, img_rows, img_cols, nchannels))
            y = tf.placeholder(tf.float32,
                               shape=(None, mc.nb_classes))

            # Define a model.
            model = mc.get_model()
            preds = model.get_logits(x)
            loss = CrossEntropy(model, smoothing=0.)

            # Define the test set accuracy evaluation.
            def evaluate():
                acc = model_eval(sess, x, y, preds, x_test, y_test,
                                 args={'batch_size': mc.batch_size})
                print('Test accuracy: %0.4f' % acc)

            # Train the model
            train_params = {'nb_epochs': mc.max_epochs,
                            'batch_size': mc.batch_size, 
                            'learning_rate': mc.learning_rate}
            
            model_dir = mc.get_model_dir_name()
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            model_filepath = os.path.join(model_dir, 'model.joblib')

            train(sess, loss, x_train, y_train, evaluate=evaluate,
              args=train_params, var_list=model.get_params())
            serial.save(model_filepath, model)


def compare_accuracies(mc, data_dict):
  # parse data_dict
  x_train = data_dict['x_train'] 
  labels_train = data_dict['labels_train']
  x_test = data_dict['x_test']
  y_test = data_dict['y_test']
  x_cali = data_dict['x_cali'] 
  labels_cali = data_dict['labels_cali']

  # Use Image Parameters.
  img_rows, img_cols, nchannels = x_train.shape[1:4]

  with mc.get_tensorflow_session() as sess:
      with tf.variable_scope('dknn'):
          # Define input TF placeholder.
          x = tf.placeholder(tf.float32,
                            shape=(None, img_rows, img_cols, nchannels))
          #y = tf.placeholder(tf.float32,
          #                  shape=(None, mc.nb_classes))
          
          model_dir = mc.get_model_dir_name()
          model = mc.load_model(model_dir=model_dir)

          # Define callable that returns a dictionary of all activations for a dataset
          def get_activations(data):
              data_activations = {}
              for layer in layers:
                  layer_sym = tf.layers.flatten(model.get_layer(x, layer))
                  data_activations[layer] = batch_eval(sess, [x], [layer_sym], [data],
                                                    args={'batch_size': mc.batch_size})[0]
              return data_activations

          # Extract representations for the training and calibration data at each layer of interest to the DkNN.
          layers = ['ReLU1', 'ReLU3', 'ReLU5', 'logits']

          #Euclidean DKNN
          dknn = DkNNModel(
          neighbors = mc.nb_neighbors,
          proto_neighbors = mc.nb_proto_neighbors,
          backend = mc.backend,
          nb_classes=mc.nb_classes,
          layers=layers,
          get_activations=get_activations,
          train_data=x_train,
          train_labels=labels_train,
          method='euclidean',
          scope='dknn')
          
          dknn.fit()
          dknn.calibrate(x_cali, labels_cali)
          preds_knn, _, _ = dknn.predict(x_test)

          # Geodesic DKNN
          dknn_geod = DkNNModel(
          neighbors = mc.nb_neighbors,
          proto_neighbors = mc.nb_proto_neighbors,
          backend = mc.backend,
          nb_classes=mc.nb_classes,
          layers=layers,
          get_activations=get_activations,
          train_data=x_train,
          train_labels=labels_train,
          method='geodesic',
          neighbors_table_path=os.path.join(mc.get_model_dir_name(),'geodesics.npy'),
          scope='dknn')
          
          dknn_geod.fit()
          dknn_geod.calibrate(x_cali, labels_cali)
          preds_geod, _, _ = dknn_geod.predict(x_test)

  dknn_acc = (preds_knn==np.argmax(y_test, axis=1)).mean()
  gdknn_acc = (preds_geod==np.argmax(y_test, axis=1)).mean()
  accuracies_dict = {'neighbors': nb_neighbors, 'DkNN': dknn_acc, 'gDkNN': gdknn_acc}
  return accuracies_dict

def get_data_dict(mc):
  # dataset = dataset_loader(config)
  dataset = dataset_loader(mc.dataset_name, mc.nb_train, mc.nb_test)

  x_train, y_train = dataset.get_set('train')
  x_test, y_test = dataset.get_set('test')
  
  # Use a holdout of the test set to simulate calibration data for the DkNN.
  labels_train = np.argmax(y_train, axis=1)
  x_cali = x_test[:mc.nb_cali]
  y_cali = y_test[:mc.nb_cali]
  labels_cali = np.argmax(y_cali, axis=1)

  y_test = y_test[mc.nb_cali:]
  x_test = x_test[mc.nb_cali:]
  labels_test = np.argmax(y_test, axis=1)

  data_dict = {'x_train': x_train,
               'y_train': y_train,
               'labels_train': labels_train,
               'x_test': x_test,
               'y_test': y_test,
               'labels_test': labels_test,
               'x_cali': x_cali,
               'y_cali': y_cali,
               'labels_cali': labels_cali}

  return data_dict

def hyperparameter_selection(mc):
  # reand and wrangle data
  data_dict = get_data_dict(mc)

  nb_neighbors_list = [128, 64, 32, 16, 8, 4, 2]
  accuracies_list = []
  for nb_neighbors in nb_neighbors_list:
    print("\n\n============ nb_neighbors:{} ============".format(nb_neighbors))
    tf.reset_default_graph()
    
    mc.nb_neighbors = nb_neighbors
    accuracies = compare_accuracies(mc, data_dict)
    accuracies_list.append(accuracies)

  model_dir = mc.get_model_dir_name()
  experiments_results_path = os.path.join(model_dir, 'accuracies.pkl')
  print('Saving Accuracies Table to {}'.format(experiments_results_path))

  accuracies_df = pd.DataFrame(accuracies_list)
  accuracies_df.to_pickle(path=experiments_results_path)


if __name__ == '__main__':
  mc = ModelConfig(config_file='../configs/config_mnist.yaml',
                   root_dir='../results/')
  #train_model(mc)
  hyperparameter_selection(mc)
