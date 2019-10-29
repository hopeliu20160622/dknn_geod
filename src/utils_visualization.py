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

#######################################################################################
# IMAGES VISUALIZATION
#######################################################################################

def mnist_transform(inp):
    inp = transforms.ToPILImage()(inp)
    return inp

def imagenet_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def cifar10_transform(inp):
    inp = transforms.ToPILImage()(inp)
    return inp

def cifar100_transform(inp):
    inp = transforms.ToPILImage()(inp)
    return inp

def plot_images(dataset_name, images, classes, labels_true, labels_pred=None):
    dataset_transforms = {'MNIST': mnist_transform,
                          'CIFAR10': cifar10_transform,
                          'CIFAR100': cifar100_transform,
                          'ImageNet': imagenet_transform}

    images = [dataset_transforms[dataset_name](image) for image in images]

    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i], interpolation='spline16')
        #ax.imshow(images[i], cmap='gray_r')

        # show true & predicted classes
        cls_true_name = classes[labels_true[i]]
        if labels_pred is None:
            #xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            xlabel = cls_true_name
        else:
            cls_pred_name = classes[labels_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

#######################################################################################
# TEXT VISUALIZATION
#######################################################################################

def print_text(sentences, vocab_stoi):
    sentence_len, n_sentences = sentences.shape
    sentence_len
    
    for idx_sentence in range(n_sentences):
        original_sentence = []
        for idx_word in range(sentence_len):
            word = int(sentences[idx_word, idx_sentence].numpy())
            original_sentence.append(vocab_stoi[word])
            if word==1 or idx_word>15:
                original_sentence = '\n'+' '.join(original_sentence)+'\n'
                break
        print(original_sentence)

#######################################################################################
# TRAJECTORIES VISUALIZATION
#######################################################################################

def plot_boostrap_trajectories(trajectories_df, suptitle, path=None):
    # colors
    num_random_seeds = trajectories_df['num_random_seeds'][0]
    colors = sns.color_palette("plasma", num_random_seeds)
    
    # data preparation
    epochs = trajectories_df.epoch.tolist()
    boostrap_trajectories = pd.DataFrame(trajectories_df.boostrap_statistics.tolist())
    
    fig, axs = plt.subplots(1, 2, figsize=(7*2, 6))
    
    for idx, random_seed in enumerate(range(num_random_seeds)):
        trajectory = pd.DataFrame(boostrap_trajectories[idx].tolist())

        axs[0].plot(epochs, trajectory['train_average_loss'],
                    label = str(random_seed), linestyle = '-', color=colors[idx])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Train Loss')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(epochs, trajectory['train_accuracy'],
                    label = str(random_seed), linestyle = '-', color=colors[idx])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Train Accuracy')
        axs[1].grid(True)
        axs[1].legend()

    fig.suptitle(suptitle, x=0.5, y=1.05)
    
    plt.savefig("../plots/"+path, bbox_inches = "tight", dpi=300)
    fig.tight_layout()
    plt.show()
