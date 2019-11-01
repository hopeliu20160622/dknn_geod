import yaml

def get_config_parameters(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    config_dict = {'max_epochs': config['max_epochs']['max_epochs'],
                   'learning_rate': config['train_parameters']['learning_rate'],
                   'lr_scheduler_step_size': config['train_parameters']['lr_scheduler_step_size'],
                   'batch_size': config['train_parameters']['nb_random_seeds'],
                   'nb_random_seeds': config['train_parameters']['nb_random_seeds'],
                   'weight_decay': config['train_parameters']['weight_decay'],
                   'nb_train': config['train_parameters']['nb_train'],
                   'nb_test': config['train_parameters']['nb_test'],
                   'nb_proto_neighbors': config['nn_parameters']['nb_proto_neighbors'],
                   'nb_neighbors': config['nn_parameters']['nb_neighbors'],
                   'nb_cali': config['nn_parameters']['nb_cali'],
                   'backend': config['nn_parameters']['backend']}
    
    return config_dict
