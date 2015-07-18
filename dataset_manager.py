import os
import sys
from config import config as c

c.dataset_dir
c.training_dir
c.test_dir


def _clear_dir(directory):
    filenames = os.listdir(directory)
    print('Removing {} files...'.format(len(filenames)))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))


def clear_training_set():
    pass


def clear_test_set():
    pass


def clear_dataset():
    clear_training_set()
    clear_dataset()


def fetch_test_set(num):
    pass


def fetch_training_set(num):
    pass


if __name__ == '__main__':
    _clear_dir('temp')
    pass
