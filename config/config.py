import yaml

# base = 'config/base.yaml'
# data = 'config/data/data.yaml'
# model = 'config/model/restnet50.yaml'
# loss = 'config/loss/ce.yaml'
# optimizer = 'config/optimizer/adam.yaml'
classification = True

base = 'config/base.yaml'
data = 'config/data/data_regression.yaml'
model = 'config/model/restnet50.yaml'
loss = 'config/loss/mse.yaml'
optimizer = 'config/optimizer/adam.yaml'
classification = False

def get_config():
    config = yaml.load(open(base, 'r'), Loader=yaml.Loader)

    config_data = yaml.load(open(data, 'r'), Loader=yaml.Loader)
    config.update(config_data)
    config_model = yaml.load(open(model, 'r'), Loader=yaml.Loader)
    config.update(config_model)
    config_loss = yaml.load(open(loss, 'r'), Loader=yaml.Loader)
    config.update(config_loss)
    config_optimizer = yaml.load(open(optimizer, 'r'), Loader=yaml.Loader)
    config.update(config_optimizer)
    return config

    