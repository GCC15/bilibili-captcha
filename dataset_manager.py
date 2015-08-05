# Manage all dataset

import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import captcha_source
import config as c
from captcha_recognizer import CaptchaRecognizer
import time

_cm_greys = plt.cm.get_cmap('Greys')
_png = '.png'

dataset_dir = c.get('dataset')
training_set_dir = os.path.join(dataset_dir, c.get('training'))
training_char_dir = os.path.join(dataset_dir, c.get('training_char'))
test_set_dir = os.path.join(dataset_dir, c.get('test'))

_PARTITION_JSON = os.path.join(dataset_dir, 'partition.json')
_partition_json = json.load(open(_PARTITION_JSON))
_FAIL = 'fail'
_SUCCESS = 'success'
_CAPTCHA_LENGTH = captcha_source.captcha_length


def _get_training_char_dir(char):
    return os.path.join(training_char_dir, char)


def _make_all_char_dirs():
    for char in captcha_source.chars:
        c.make_dirs(_get_training_char_dir(char))


c.make_dirs(training_set_dir)
c.make_dirs(training_char_dir)
c.make_dirs(test_set_dir)
_make_all_char_dirs()


def char_path(char, path):
    return os.path.join(training_char_dir, char, path)


# Fetch some CAPTCHA images from a CAPTCHA source to a directory
def _fetch_captchas_to_dir(directory, num=1, use_https=False):
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
        while len(seq) != _CAPTCHA_LENGTH:
            print('Incorrect length, length should be {0}'.format(_CAPTCHA_LENGTH))
            seq = input('[{}] Enter the char sequence: '.format(i))
            seq = captcha_source.canonicalize(seq)
        path = os.path.join(directory, _add_suffix(seq))
        if not os.path.isfile(path):
            mpimg.imsave(path, img)
        else:
            print('Warning: char sequence already exists in dataset! Skipping')
    plt.ioff()


def clear_training_set():
    c.clear_dir(training_set_dir)


def clear_training_chars():
    for directory in os.listdir(training_char_dir):
        c.clear_dir(os.path.join(training_char_dir, directory))


def clear_test_set():
    c.clear_dir(test_set_dir)


# def clear_dataset():
#     clear_training_set()
#     clear_test_set()


def fetch_training_set(num=1, use_https=False):
    _fetch_captchas_to_dir(training_set_dir, num, use_https)


def fetch_test_set(num=1, use_https=False):
    _fetch_captchas_to_dir(test_set_dir, num, use_https)


# Get one image from a directory
def _get_image(directory, filename):
    image = mpimg.imread(os.path.join(directory, filename))
    # Discard alpha channel
    return image[:, :, 0:3]


# Get some images from a directory
def _get_images(directory, num=1):
    filenames = _list_png(directory)
    if num > len(filenames):
        num = len(filenames)
        print('Warning: requesting more images than stored, returning all '
              'available')
    else:
        random.shuffle(filenames)
    return [_get_image(directory, filenames[i]) for i in range(num)]


def _add_suffix(basename, suffix=_png):
    return '{}{}'.format(basename, suffix)


def _remove_suffix(filename):
    basename, ext = os.path.splitext(filename)
    return basename


def _get_suffix(filename):
    basename, ext = os.path.splitext(filename)
    return ext


def get_test_image(seq):
    return _get_image(test_set_dir, _add_suffix(seq))


def get_test_images(num=1):
    return _get_images(test_set_dir, num)


# Return a training image randomly if seq is None
def get_training_image(seq=None):
    if seq is None:
        return get_training_images(1)[0]
    else:
        return _get_image(training_set_dir, _add_suffix(seq))


def get_training_images(num=1):
    return _get_images(training_set_dir, num)


# List all png files in a directory
def _list_png(directory):
    def png_filter(filename):
        return _get_suffix(filename) == _png

    return list(filter(png_filter, os.listdir(directory)))


# def _list_unconverted_png(directory):
#     list = _list_png(directory)
#     final_list = []
#     for i in range(len(list)):
#         if list[i][5:9] != ".con":
#             final_list.append(list[i])
#     return final_list


# def _list_seq(directory):
#     return list(map(_remove_suffix, _list_png(directory)))


# def _list_unconverted_seq(directory):
#     return list(
#         map(_remove_suffix, _list_unconverted_png(directory)))


# def _convert_all_training_png_to_original_name():
#     png_list = _list_png(c.training_set_dir)
#     for i in range(len(png_list)):
#         os.rename(os.path.join(c.training_set_dir, png_list[i]),
#                   os.path.join(c.training_set_dir, (png_list[i][0:5] + '.png')))


def partition_training_images_to_chars(force_update=False):
    time_start = time.time()
    fail_list = [] if force_update else _partition_json[_FAIL]
    success_list = [] if force_update else _partition_json[_SUCCESS]
    old_file_set = set(fail_list + success_list)

    def filename_filter(f):
        return f not in old_file_set

    filenames = _list_png(training_set_dir)
    num_total = len(filenames)
    filenames = list(filter(filename_filter, filenames))
    num_update = len(filenames)
    num_success = 0
    recognizer = CaptchaRecognizer()
    for n in range(num_update):
        filename = filenames[n]
        seq = _remove_suffix(filename)
        print('{}/{}: {}'.format(n, num_update, seq))
        img = get_training_image(seq)
        img_01 = recognizer.remove_noise_with_hsv(img)
        img_02 = recognizer.remove_noise_with_neighbors(img_01)
        img_02 = recognizer.remove_noise_with_neighbors(img_02)
        _, cut_line = recognizer.find_vertical_separation_line(img_02)
        img_list = recognizer.cut_images_by_vertical_line(img_02, cut_line)
        if len(img_list) == captcha_source.captcha_length:
            success_list.append(filename)
            num_success += 1
            for i in range(len(img_list)):
                path = char_path(seq[i], _add_suffix('{}.{}'.format(seq, i + 1)))
                mpimg.imsave(path, img_list[i], cmap=_cm_greys)
        else:
            fail_list.append(filename)
    success_list.sort()
    fail_list.sort()
    json.dump(
        {
            _FAIL: fail_list,
            _SUCCESS: success_list
        },
        open(_PARTITION_JSON, 'w'),
        indent=2
    )
    print('Total: {}'.format(num_total))
    print('Update: {}'.format(num_update))
    print('Success: {}'.format(num_success))
    time_end = time.time()
    print('Elapsed time: {}'.format(time_end - time_start))


if __name__ == '__main__':
    pass
