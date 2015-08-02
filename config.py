# Configurations and file system related operations

import os
import json

CONFIG_JSON = 'config.json'
print('Loading {}'.format(CONFIG_JSON))
config_dict = json.load(open(CONFIG_JSON))
chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')


# Create a directory if not exist
def _make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _make_all_char_dirs():
    for i in chars:
        _make_dirs(os.path.join(training_char_dir, i))


dataset_dir = config_dict['dataset']
training_set_dir = os.path.join(dataset_dir, config_dict['training'])
training_char_dir = os.path.join(dataset_dir, config_dict['training_char'])
test_set_dir = os.path.join(dataset_dir, config_dict['test'])
temp_dir = config_dict['temp']

_make_dirs(dataset_dir)
_make_dirs(training_set_dir)
_make_dirs(training_char_dir)
_make_dirs(test_set_dir)
_make_dirs(temp_dir)
_make_all_char_dirs()


def temp_path(filename):
    return os.path.join(temp_dir, filename)


def char_path(char, filename):
    path = os.path.join(training_char_dir, char)
    return os.path.join(path, filename)


def clear_dir(directory):
    filenames = os.listdir(directory)
    print('Removing {} files...'.format(len(filenames)))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))


def clear_temp():
    clear_dir(temp_dir)
