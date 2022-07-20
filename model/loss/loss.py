import torch.nn as nn

mapping = {
    'CE':nn.CrossEntropyLoss,
    'MSE':nn.MSELoss
}
def get_loss_from_config(config, class_weight=None):
    if 'class_weight' in config and config['class_weight']:
        return mapping[config['name']](weight=class_weights)
    else:
        return mapping[config['name']]()
