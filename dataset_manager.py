# Manages all dataset

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import captcha_source
import config as c
from captcha_recognizer import CaptchaRecognizer

_cm_greys = plt.cm.get_cmap('Greys')
_png = '.png'


def _fetch_dir(directory, num=1, use_https=False):
    plt.ion()
    plt.show()
    for i in range(num):
        img = captcha_source.fetch_image(use_https)
        plt.clf()
        plt.axis('off')
        plt.imshow(img)
        # http://stackoverflow.com/questions/12670101/matplotlib-ion-function
        # -fails-to-be-interactive
        # https://github.com/matplotlib/matplotlib/issues/1646/
        plt.show()
        plt.pause(1e-2)
        seq = input('[{}] Enter the char sequence: '.format(i))
        seq = captcha_source.canonicalize(seq)
        if len(seq) != captcha_source.captcha_length:
            raise ValueError('Incorrect length')
        # Assume the char sequence is not met before
        mpimg.imsave(os.path.join(directory, _get_filename_from_basename(seq)), img)
    plt.ioff()


def clear_training_set():
    c.clear_dir(c.training_set_dir)


def clear_training_chars():
    c.clear_dir(c.training_char_dir)


def clear_test_set():
    c.clear_dir(c.test_set_dir)


def clear_dataset():
    clear_training_set()
    clear_test_set()


def fetch_training_set(num=1, use_https=False):
    _fetch_dir(c.training_set_dir, num, use_https)


def fetch_test_set(num=1, use_https=False):
    _fetch_dir(c.test_set_dir, num, use_https)


def _get_image(directory, filename):
    image = mpimg.imread(os.path.join(directory, filename))
    # Discard alpha channel
    return image[:, :, 0:3]


def _get_images(directory, num=1):
    images = []
    filenames = _list_png(directory)
    if num > len(filenames):
        num = len(filenames)
        print(
            'Requesting more images than stored, returning all available '
            'images')
    else:
        random.shuffle(filenames)
    for i in range(num):
        images.append(_get_image(directory, filenames[i]))
    return images


def _get_basename_from_filename(filename):
    basename, ext = os.path.splitext(filename)
    return basename


def _get_filename_from_basename(basename):
    return '{}{}'.format(basename, _png)


def get_test_image(seq):
    return _get_image(c.test_set_dir, _get_filename_from_basename(seq))


def get_test_images(num=1):
    return _get_images(c.test_set_dir, num)


def get_training_image(seq):
    return _get_image(c.training_set_dir, _get_filename_from_basename(seq))


def get_training_images(num=1):
    return _get_images(c.training_set_dir, num)


# List all png files in a directory
def _list_png(directory):
    def png_filter(filename):
        root, ext = os.path.splitext(filename)
        return ext == _png

    return list(filter(png_filter, os.listdir(directory)))


def _list_seq(directory):
    return list(map(_get_basename_from_filename, _list_png(directory)))


def convert_training_image_to_char(force_update=False):
    num_total = 0
    num_update = 0
    num_success = 0
    recognizer = CaptchaRecognizer()
    seq_list = _list_seq(c.training_set_dir)
    for s in range(len(seq_list)):
        num_total += 1
        seq = seq_list[s]
        print('{}/{}: {}'.format(s, len(seq_list), seq))
        if force_update or not os.path.isfile(
                c.char_path(seq[0],
                _get_filename_from_basename('{}.{}'.format(seq, 1)))):
            num_update += 1
            img = get_training_image(seq)
            img_01 = recognizer.remove_noise_with_hsv(img)
            img_02 = recognizer.remove_noise_with_neighbors(img_01)
            img_02 = recognizer.remove_noise_with_neighbors(img_02)
            _, cut_line = recognizer.find_vertical_separation_line(img_02)
            img_list = recognizer.cut_images(img_02, cut_line)
            if len(img_list) == captcha_source.captcha_length:
                num_success += 1
                for i in range(captcha_source.captcha_length):
                    path = c.char_path(
                        seq[i],
                        _get_filename_from_basename('{}.{}'.format(seq, i + 1)))
                    mpimg.imsave(path, img_list[i], cmap=_cm_greys)
    print('Total: {}'.format(num_total))
    print('Update: {}'.format(num_update))
    print('Success: {}'.format(num_success))


if __name__ == '__main__':
    pass
