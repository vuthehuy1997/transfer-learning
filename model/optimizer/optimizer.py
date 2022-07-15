from torch import optim

mapping = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'Adagrad': optim.Adagrad,
    'RMSprop': optim.RMSprop,
}
def get_optimizer_from_config(config,params):
    optim_name = config['name']
    optimizer = mapping[optim_name](
        params=params,
        **config['args'])
    return optimizer