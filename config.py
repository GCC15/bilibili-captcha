import os
import json

CONFIG_JSON = 'config.json'
print('Loading {}'.format(CONFIG_JSON))
config_dict = json.load(open(CONFIG_JSON))


# Create a directory if not exist
def _make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


dataset_dir = config_dict['dataset']
training_set_dir = os.path.join(dataset_dir, config_dict['training'])
test_set_dir = os.path.join(dataset_dir, config_dict['test'])
temp_dir = config_dict['temp']

_make_dirs(dataset_dir)
_make_dirs(training_set_dir)
_make_dirs(test_set_dir)
_make_dirs(temp_dir)


def temp_path(filename):
    return os.path.join(temp_dir, filename)
