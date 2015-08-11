# Manage all dataset

import os
import random
import json
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from captcha_source import CaptchaSource
import config as c
from captcha_recognizer import CaptchaRecognizer
import helper

_cm_greys = plt.cm.get_cmap('Greys')
_png = '.png'

_dataset_dir = c.get('dataset')
_training_set_dir = os.path.join(_dataset_dir, c.get('training'))
_training_char_dir = os.path.join(_dataset_dir, c.get('training_char'))
_test_set_dir = os.path.join(_dataset_dir, c.get('test'))

_PARTITION_JSON = os.path.join(_dataset_dir, 'partition.json')
_NUM_TOTAL = '###total'
_NUM_FAIL = '##fail'
_NUM_SUCCESS = '##success'
_SUCCESS_RATE = '##success_rate'
_NUM_CHAR = '#{}'
_FAIL = 'fail'
_SUCCESS = 'success'

_SEQ_SKIP = '0'
_source = CaptchaSource()


def _contain_invalid_char(seq):
    return any(char not in _source.charset for char in seq)


def _get_training_char_dir(char):
    return os.path.join(_training_char_dir, char)


c.make_dirs(_training_set_dir)
c.make_dirs(_training_char_dir)
c.make_dirs(_test_set_dir)
for _char in _source.chars:
    c.make_dirs(_get_training_char_dir(_char))


def _get_training_char_path(char, path):
    return os.path.join(_get_training_char_dir(char), path)


# Fetch some CAPTCHA images from a CAPTCHA source to a directory
def _fetch_captchas_to_dir(directory, num=1):
    plt.ion()
    plt.show()
    for i in range(num):
        img = _source.fetch_captcha()
        plt.clf()
        plt.axis('off')
        plt.imshow(img)
        # http://stackoverflow.com/questions/12670101/matplotlib-ion-function
        # -fails-to-be-interactive
        # https://github.com/matplotlib/matplotlib/issues/1646/
        plt.show()
        plt.pause(1e-2)

        while True:
            seq = input('[{}] Enter the char sequence: '.format(i + 1))
            # To skip a CAPTCHA.
            # Warning: skipping may reduce the quality of the training set.
            if seq == _SEQ_SKIP:
                break
            seq = helper.canonicalize(seq)
            if (len(seq) != _source.captcha_length or
                    _contain_invalid_char(seq)):
                print('Invalid sequence!')
            else:
                break
        if seq == _SEQ_SKIP:
            print('Skipped manually')
            continue
        path = os.path.join(directory, _add_suffix(seq))
        if os.path.isfile(path):
            print('Warning: char sequence already exists in dataset! Skipping')
        else:
            mpimg.imsave(path, img)
    plt.ioff()


def clear_training_set():
    pass  # for safety reason
    # c.clear_dir(_training_set_dir)


def clear_training_chars():
    for directory in (os.listdir(_training_char_dir)):
        if directory in _source.chars:
            c.clear_dir(os.path.join(_training_char_dir, directory))


def clear_test_set():
    c.clear_dir(_test_set_dir)


# def clear_dataset():
#     clear_training_set()
#     clear_test_set()


def fetch_training_set(num=1):
    _fetch_captchas_to_dir(_training_set_dir, num)


def fetch_test_set(num=1):
    _fetch_captchas_to_dir(_test_set_dir, num)


# Get one image from a directory
def _get_image(directory, filename, mode='rgb'):
    image = mpimg.imread(os.path.join(directory, filename))
    if mode == 'rgb':
        return image[:, :, 0:3]
    elif mode == 'gray':
        return 1 - image[:, :, 0]


# Get some images from a directory
# set num = 0 or None to get all
def _get_images(directory, num=None, mode='rgb'):
    filenames = _list_png(directory)
    if num:
        if num > len(filenames):
            num = len(filenames)
            print('Warning: requesting more images than stored, returning all '
                  'available')
        else:
            random.shuffle(filenames)
    else:
        num = len(filenames)
    return [_get_image(directory, filenames[i], mode) for i in range(num)]


def _add_suffix(basename, suffix=_png):
    return '{}{}'.format(basename, suffix)


def _remove_suffix(filename):
    basename, ext = os.path.splitext(filename)
    return basename


def _get_suffix(filename):
    basename, ext = os.path.splitext(filename)
    return ext


# def get_test_image(seq):
#     return _get_image(test_set_dir, _add_suffix(seq))


# def get_test_images(num=1):
#     return _get_images(test_set_dir, num)


# Return a training image randomly if seq is None
def get_training_image(seq=None):
    if seq is None:
        return get_training_images(1)[0]
    else:
        return _get_image(_training_set_dir, _add_suffix(seq))


def get_training_images(num=None):
    return _get_images(_training_set_dir, num)


def get_training_char_images(char, num=None):
    return _get_images(_get_training_char_dir(char), num, mode='gray')


# List all png files in a directory
def _list_png(directory):
    def png_filter(filename):
        return _get_suffix(filename) == _png

    return list(filter(png_filter, os.listdir(directory)))


def _list_basename(directory):
    return list(map(_remove_suffix, _list_png(directory)))


def partition_training_images_to_chars(captcha_recognizer=CaptchaRecognizer(),
                                       force_update=False,
                                       save=True):
    time_start = time.time()
    try:
        json_dict = json.load(open(_PARTITION_JSON))
    except Exception as e:
        print(e)
        print('Warning: failed to load {}. Reconstructing...'.
              format(_PARTITION_JSON))
        json_dict = {}
        force_update = True
    if force_update:
        json_dict[_FAIL] = []
        json_dict[_SUCCESS] = []
        for char in _source.chars:
            json_dict[_NUM_CHAR.format(char)] = 0
    seqs = _list_basename(_training_set_dir)
    num_total = len(seqs)
    old_seq_set = set(json_dict[_FAIL] + json_dict[_SUCCESS])

    def seq_filter(s):
        return s not in old_seq_set

    seqs = list(filter(seq_filter, seqs))
    num_update = len(seqs)
    num_update_success = 0
    recognizer = captcha_recognizer

    for n in range(num_update):
        seq = seqs[n]
        if save:
            print('{}/{}: {}'.format(n, num_update, seq))
        img = get_training_image(seq)
        char_images = recognizer.partition(img)
        # If successful
        if char_images is not None:
            json_dict[_SUCCESS].append(seq)
            num_update_success += 1
            for i in range(len(char_images)):
                char = seq[i]
                json_dict[_NUM_CHAR.format(char)] += 1
                if save:
                    path = _get_training_char_path(char, _add_suffix(
                        '{}.{}'.format(seq, i + 1)))
                    mpimg.imsave(path, char_images[i], cmap=_cm_greys)
        else:
            json_dict[_FAIL].append(seq)

    num_total_success = len(json_dict[_SUCCESS])
    json_dict[_NUM_TOTAL] = num_total
    json_dict[_NUM_FAIL] = num_total - num_total_success
    json_dict[_NUM_SUCCESS] = num_total_success
    total_success_rate = num_total_success / num_total if num_total else 0
    json_dict[_SUCCESS_RATE] = '{:.3%}'.format(total_success_rate)
    json_dict[_FAIL].sort()
    json_dict[_SUCCESS].sort()
    json.dump(
        json_dict,
        open(_PARTITION_JSON, 'w'),
        sort_keys=True,
        indent=2
    )
    if save:
        print('Update: {}'.format(num_update))
        print('Update success: {}'.format(num_update_success))
        if num_update:
            print('Update success rate is: {}'.format(
                num_update_success / num_update))
        print('Total: {}'.format(num_total))
        print('Total success: {}'.format(num_total_success))
        print('Total success rate is: {}'.format(total_success_rate))
        time_end = time.time()
        print('Elapsed time of partitioning training images: {}'.format(
            time_end - time_start))
    if not save:
        print('h_tol = {}'.format(recognizer.h_tolerance))
        print('s_tol = {}'.format(recognizer.s_tolerance))
        print('v_tol = {}'.format(recognizer.v_tolerance))
        print('Total success rate is: {}'.format(total_success_rate))
    return total_success_rate


# The best parameter combination now is (6,36,40)
def tune_partition_parameter():
    h_tol = [6]
    s_tol = [36]
    v_tol = [40]
    rate = np.zeros((len(h_tol), len(s_tol), len(v_tol)))
    for h in h_tol:
        for s in s_tol:
            for v in v_tol:
                recognizer = CaptchaRecognizer(_source, h / 360, s / 100,
                                               v / 100)
                rate[
                    h - 6, s - 36, v - 40] = \
                    partition_training_images_to_chars(recognizer,
                                                       force_update=True,
                                                       save=False)
    print(np.unravel_index(rate.argmax(), rate.shape))
    print(rate.max())
    np.save(os.path.join(_dataset_dir, 'gridsearch.npy'), rate)
    return rate
