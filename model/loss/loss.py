import torch.nn as nn

mapping = {
    'ce':nn.CrossEntropyLoss
}
def get_optimizer_from_config(name):
    return mapping[name]
