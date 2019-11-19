import os
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


def compare_accuracies(nb_neighbors):
  acc1 = 1
  acc2 = 2
  accuracies_dict = {'neighbors': nb_neighbors, 'DkNN': acc1, 'gDkNN':acc2}
  return accuracies_dict

def hyperparameter_selection(mc):
  nb_neighbors_list = [2, 4, 8, 16, 32, 64, 128]

  accuracies_list = []

  # Instantiate and fit DkNN and gDkNN

  for nb_neighbors in nb_neighbors_list:
    print("nb_neighbors", nb_neighbors)
    accuracies = compare_accuracies(nb_neighbors)
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
