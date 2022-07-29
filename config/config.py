import yaml

def get_config(config_file):
    config_config = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)

    config = yaml.load(open(config_config['base'], 'r'), Loader=yaml.Loader)

    config_data = yaml.load(open(config_config['data'], 'r'), Loader=yaml.Loader)
    config.update(config_data)
    config_model = yaml.load(open(config_config['model'], 'r'), Loader=yaml.Loader)
    config.update(config_model)
    config_loss = yaml.load(open(config_config['loss'], 'r'), Loader=yaml.Loader)
    config.update(config_loss)
    config_optimizer = yaml.load(open(config_config['optimizer'], 'r'), Loader=yaml.Loader)
    config.update(config_optimizer)

    config['classification'] = config_config['classification']
    config['aug'] = config_config['aug']
    config['device'] = config_config['device']
    return config

    