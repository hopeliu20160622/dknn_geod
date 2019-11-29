import os
import time
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
    
    with mc.get_tensorflow_session() as sess:
        with tf.variable_scope('dknn'):
            # Define input TF placeholder.
            x = tf.placeholder(tf.float32,
                               shape=(None, mc.img_rows, mc.img_cols, mc.nchannels))
            y = tf.placeholder(tf.float32,
                               shape=(None, mc.nb_classes))

            # Define a model.
            model = mc.get_model(scope='dknn')
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
                            'learning_rate': mc.learning_rate,
                            'loss_threshold': mc.loss_threshold}

            model_dir = mc.get_model_dir_name()
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            model_filepath = os.path.join(model_dir, 'model.joblib')

            train(sess, loss, x_train, y_train, evaluate=evaluate,
              args=train_params, var_list=model.get_params())
            serial.save(model_filepath, model)


def compare_accuracies(mc, data_dict, nb_neighbors_list):
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
          
          model_dir = mc.get_model_dir_name()
          model = mc.load_model(model_dir=model_dir)

          # Extract representations for the training and calibration data at each layer of interest to the DkNN.
          if mc.dataset_name=='MNIST' or mc.dataset_name=='SVHN':
            layers = ['ReLU1', 'ReLU3', 'ReLU5', 'logits']
          elif mc.dataset_name=='CIFAR10':
            layers = ['Input0', 'Conv2D1', 'Flatten2', 'logits']

          #Euclidean DKNN
          dknn = DkNNModel(
          sess=sess,
          model=model,
          backend=mc.backend,
          neighbors=mc.nb_neighbors,
          proto_neighbors=mc.nb_proto_neighbors,
          img_rows=mc.img_rows,
          img_cols=mc.img_cols,
          nchannels=mc.nchannels,
          nb_classes=mc.nb_classes,
          layers=layers,
          train_data=x_train,
          train_labels=labels_train,
          method='euclidean',
          neighbors_table_path=mc.get_model_dir_name(),
          scope='dknn')
          
          start = time.time()
          dknn.fit()
          print("Fit time: {}".format(time.time()-start))

          # Geodesic DKNN
          dknn_geod = DkNNModel(
          sess=sess,
          model=model,
          backend=mc.backend,
          neighbors=mc.nb_neighbors,
          proto_neighbors=mc.nb_proto_neighbors,
          img_rows=mc.img_rows,
          img_cols=mc.img_cols,
          nchannels=mc.nchannels,
          nb_classes=mc.nb_classes,
          layers=layers,
          train_data=x_train,
          train_labels=labels_train,
          method='geodesic',
          neighbors_table_path=mc.get_model_dir_name(),
          scope='dknn')
          
          start = time.time()
          dknn_geod.fit()
          print("Fit time: {}".format(time.time()-start))

          test_activations = dknn_geod.get_activations(x_test)
          test_softmax = test_activations['logits']
          baseline_accuracy = (np.argmax(test_softmax,axis=1)==np.argmax(y_test,axis=1)).mean()
          print('Baseline accuracy: {}'.format(baseline_accuracy))

          accuracies_list = []
          for nb_neighbors in nb_neighbors_list:
            print("\n\n============ nb_neighbors:{} ============".format(nb_neighbors))
            start_time = time.time()
            mc.nb_neighbors = nb_neighbors
            dknn.update_neighbors(mc.nb_neighbors)
            dknn_geod.update_neighbors(mc.nb_neighbors)
            
            start = time.time()
            dknn.calibrate(x_cali, labels_cali)
            preds_knn, _, _ = dknn.predict(x_test)
            print("Calibrate and predict time: {}".format(time.time()-start))
            
            start = time.time()
            dknn_geod.calibrate(x_cali, labels_cali)
            preds_geod, _, _ = dknn_geod.predict(x_test)
            print("Calibrate and predict time: {}".format(time.time()-start))

            dknn_acc = (preds_knn==np.argmax(y_test, axis=1)).mean()
            gdknn_acc = (preds_geod==np.argmax(y_test, axis=1)).mean()
            accuracies_dict = {'neighbors': mc.nb_neighbors, 'Baseline':baseline_accuracy,'DkNN': dknn_acc, 'gDkNN': gdknn_acc}

            accuracies_list.append(accuracies_dict)
            print(accuracies_dict)
            print("Neighbors time: {}".format(time.time()-start_time))

  return accuracies_list

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

  nb_neighbors_list = [512, 256, 128, 64, 32, 16]
  mc.nb_neighbors = max(nb_neighbors_list)

  accuracies_list = compare_accuracies(mc, data_dict, nb_neighbors_list)

  model_dir = mc.get_model_dir_name()
  experiments_results_path = os.path.join(model_dir, 'accuracies.pkl')
  print('Saving Accuracies Table to {}'.format(experiments_results_path))

  accuracies_df = pd.DataFrame(accuracies_list)
  accuracies_df.to_pickle(path=experiments_results_path)

if __name__ == '__main__':
  mc = ModelConfig(config_file='../configs/config_svhn.yaml',
                   root_dir='../results/')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(mc.gpu_device)

  #train_model(mc)
  hyperparameter_selection(mc)
