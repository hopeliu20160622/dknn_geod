import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

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

#######################################################################################
# DISTRIBUTIONS
#######################################################################################

def plot_distributions(distributions_dict, xlabel, fig_title):
    n_distributions = len(distributions_dict.keys())
    fig, ax = plt.subplots(1, figsize=(7, 5.5))
    plt.subplots_adjust(wspace=0.35)
    
    colors = sns.color_palette("plasma", len(distributions_dict.keys()))
    
    for idx, dist_name in enumerate(distributions_dict.keys()):
        train_dist_plot = sns.kdeplot(distributions_dict[dist_name],
                                      bw='silverman',
                                      label=dist_name,
                                      color=colors[idx])
    
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(fig_title, fontsize=15.5)
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    
    fig.tight_layout()
    plt.savefig("./results/distributions.png", bbox_inches = "tight", dpi=300)
    plt.show()
