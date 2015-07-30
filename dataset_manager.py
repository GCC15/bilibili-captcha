import os
from config import config as c


def _clear_dir(directory):
    filenames = os.listdir(directory)
    print('Removing {} files...'.format(len(filenames)))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))


def _fetch_dir():
    pass


def clear_training_set():
    _clear_dir(c.training_set_dir)


def clear_test_set():
    _clear_dir(c.test_set_dir)


def clear_dataset():
    clear_training_set()
    clear_test_set()


def fetch_test_set(num):
    pass


def fetch_training_set(num):
    pass


if __name__ == '__main__':
    _clear_dir(c.temp_dir)
    pass
