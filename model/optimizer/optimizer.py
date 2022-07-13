from torch import optim

def get_optimizer_from_config(config,params):
    optim_name = config['optimizer']['name']

    if optim_name == 'SGD':
        optimizer = optim.SGD(
            params=params),
            lr=config['optimizer']['lr'],
            momentum=['optimizer']['momentum'])
    elif optim_name == 'Adam':
        optimizer = optim.Adam(
            params=params,
            lr=config['optimizer']['start_lr'],
            weight_decay=config['optimizer']['weight_decay'])
    