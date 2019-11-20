from cleverhans.dataset import Dataset
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import urllib.request
import shutil
import os
from scipy import io
import numpy as np


class SVHN(Dataset):
#   """The SVHN dataset"""
    def __init__(self, train_start=0, train_end=73257, test_start=0, test_end=26032,
                center=False, max_val=1.):
        kwargs = locals()
        if '__class__' in kwargs:
            del kwargs['__class__']
        if kwargs is None:
            kwargs = {}
        if "self" in kwargs:
            del kwargs["self"]
        self.kwargs = kwargs
        self.PATH='../data/svhn'
        packed = self.data_svhn(train_start=train_start,
                              train_end=train_end,
                              test_start=test_start,
                              test_end=test_end)
        x_train, y_train, x_test, y_test = packed
        
        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.
        x_train *= max_val
        x_test *= max_val
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_val = max_val
        
    def to_tensorflow(self, shuffle=4096):
    # This is much more efficient with data augmentation, see tutorials.
        return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
            self.in_memory_dataset(self.x_test, self.y_test, repeat=False))
    
    def data_svhn(self, train_start=0, train_end=73257, test_start=0, test_end=26032):
        """
        Preprocess SVHN dataset
        :return:
        """
        # These values are specific to SVHN
        img_rows = 32
        img_cols = 32
        nb_classes = 10

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
#         if tf.keras.backend.image_data_format() == 'channels_first':
#             x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
#             x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
#         else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        # convert class vectors to binary class matrices
        y_train = self.to_categorical(y_train, nb_classes)
        y_test = self.to_categorical(y_test, nb_classes)
        x_train = x_train[train_start:train_end, :, :, :]
        y_train = y_train[train_start:train_end, :]
        x_test = x_test[test_start:test_end, :]
        y_test = y_test[test_start:test_end, :]

        return x_train, y_train, x_test, y_test
    
    def load_data(self, ):
        if not os.path.isdir(self.PATH):
            os.makedirs(self.PATH)
        train_file = os.path.join(self.PATH, 'train_32x32.mat')
        test_file = os.path.join(self.PATH, 'test_32x32.mat')
        if not os.path.exists(train_file):
            url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
            print('downloading SVHN training data to {}'.format(train_file))
            with urllib.request.urlopen(url) as response, open(train_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        if not os.path.exists(test_file):
            url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
            print('downloading SVHN test data to {}'.format(test_file))
            with urllib.request.urlopen(url) as response, open(test_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        train = io.loadmat(train_file)
        test = io.loadmat(test_file)
        
        train = (np.transpose(train['X'], (3,0,1,2)), train['y'])
        test = (np.transpose(test['X'], (3,0,1,2)), test['y'])
        
        return train, test
    
    def to_categorical(self, y, nb_classes):
        res = np.zeros((y.shape[0], nb_classes))
        y = np.where(y==10, 0, y)
        res[np.arange(res.shape[0]), y.reshape(-1)] = 1
        return res
