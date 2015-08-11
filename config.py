# Configurations (e.g. paths), and generic file system operations
# Use me: import config as c

import os
import json

_CONFIG_JSON = 'config.json'
_json = json.load(open(_CONFIG_JSON))


def get(key):
    return _json[key]


_temp_dir = get('temp')


# Create a directory and its parents, if necessary
def make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


make_dirs(_temp_dir)


def temp_path(path):
    return os.path.join(_temp_dir, path)


def clear_dir(directory):
    filenames = os.listdir(directory)
    print('Removing {} files...'.format(len(filenames)))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))


def clear_temp():
    clear_dir(_temp_dir)
