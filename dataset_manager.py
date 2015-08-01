# Manages all dataset

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import captcha_source
import config as c
from PIL import Image


def _clear_dir(directory):
    filenames = os.listdir(directory)
    print('Removing {} files...'.format(len(filenames)))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))


def _fetch_dir(directory, num=1, use_https=False):
    plt.ion()
    plt.show()
    for _ in range(num):
        img = captcha_source.fetch_image(use_https)
        plt.clf()
        plt.axis('off')
        plt.imshow(img)
        # http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive
        # https://github.com/matplotlib/matplotlib/issues/1646/
        plt.show()
        plt.pause(1e-2)
        seq = input('Enter the char sequence: ')
        seq = captcha_source.canonicalize(seq)
        if len(seq) != captcha_source.captcha_length:
            raise ValueError('Incorrect length')
        # Assume the char sequence is not met before
        mpimg.imsave(os.path.join(directory, '{}.png'.format(seq)), img)
    plt.ioff()


def clear_training_set():
    _clear_dir(c.training_set_dir)


def clear_test_set():
    _clear_dir(c.test_set_dir)


def clear_dataset():
    clear_training_set()
    clear_test_set()


def fetch_training_set(num=1, use_https=False):
    _fetch_dir(c.training_set_dir, num)


def fetch_test_set(num=1, use_https=False):
    _fetch_dir(c.test_set_dir, num)


def _get_image(directory, filename):
    image = mpimg.imread(os.path.join(directory, filename))
    return image[:, :, 0:3]


def _get_images(directory, num=1):
    images = []
    filenames = _list_png(directory)
    if num > len(filenames):
        num = len(filenames)
        print('Requesting more images than stored, returning all available images')
    else:
        random.shuffle(filenames)
    for i in range(num):
        images.append(_get_image(directory, filenames[i]))
    return images


def get_test_image(seq):
    return _get_image(c.test_set_dir, '{}.png'.format(seq))


def get_test_images(num=1):
    return _get_images(c.test_set_dir, num)


def get_training_image(seq):
    return _get_image(c.training_set_dir, '{}.png'.format(seq))


def get_training_images(num=1):
    return _get_images(c.training_set_dir, num)


# List all png files in a directory
def _list_png(directory):
    def png_filter(filename):
        root, ext = os.path.splitext(filename)
        return ext == '.png'

    return list(filter(png_filter, os.listdir(directory)))


if __name__ == '__main__':
    pass
