import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#import seaborn as sns

#from utils_experiment import parse_trajectory_layer_statistics

#######################################################################################
# KERNEL VISUALIZATION
#######################################################################################

def plot_kernel(kernel):
    N = len(kernel)
    plt.pcolormesh(kernel)
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.show()
