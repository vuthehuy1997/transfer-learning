import torch.nn as nn

mapping = {
    'CE':nn.CrossEntropyLoss
}
def get_loss_from_config(config, class_weight=None):
    if config['class_weight']:
        return mapping[config['name']](weight=class_weights)
    else:
        return mapping[config['name']]()
