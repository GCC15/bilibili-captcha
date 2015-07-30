import os
import json

CONFIG_JSON = 'config.json'
print('Loading {}'.format(CONFIG_JSON))
config_dict = json.load(open(CONFIG_JSON))


class Config:
    pass


config = Config()
config.dataset_dir = config_dict['dataset']
config.training_set_dir = os.path.join(config.dataset_dir, config_dict['training'])
config.test_set_dir = os.path.join(config.dataset_dir, config_dict['test'])
config.temp_dir = config_dict['temp']
