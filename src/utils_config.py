import yaml
from cleverhans.dataset import MNIST, CIFAR10
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear, Softmax


def get_config_parameters(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    config_dict = {'dataset_name': config['dataset_name'],
                   'max_epochs': config['train_parameters']['max_epochs'],
                   'learning_rate': config['train_parameters']['learning_rate'],
                   'lr_scheduler_step_size': config['train_parameters']['lr_scheduler_step_size'],
                   'batch_size': config['train_parameters']['nb_random_seeds'],
                   'nb_random_seeds': config['train_parameters']['nb_random_seeds'],
                   'weight_decay': config['train_parameters']['weight_decay'],
                   'nb_train': config['train_parameters']['nb_train'],
                   'nb_test': config['train_parameters']['nb_test'],
                   'gpu_memory_fraction': config['train_parameters']['gpu_memory_fraction'],
                   'nb_proto_neighbors': config['nn_parameters']['nb_proto_neighbors'],
                   'nb_neighbors': config['nn_parameters']['nb_neighbors'],
                   'nb_cali': config['nn_parameters']['nb_cali'],
                   'backend': config['nn_parameters']['backend']}
    
    return config_dict


def dataset_loader(config):
    datasets_parser = {'MNIST':MNIST,
                       'CIFAR10':CIFAR10}
    data_loader = datasets_parser[config['dataset_name']]
    dataset = data_loader(train_start=0, train_end=config['nb_train'], 
                          test_start=0, test_end=config['nb_test'])
    return dataset


def get_model(config):
    """The model for the picklable models tutorial.
    """
    if config['dataset_name'] == 'MNIST':
        nb_filters=64
        nb_classes=10
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
    elif config['dataset_name'] == 'CIFAR10':
        nb_filters=64
        nb_classes=10
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
