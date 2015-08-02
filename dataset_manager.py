# Manages all dataset

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import captcha_source
import config as c
import captcha_recognizer as caprec

_cm_greys = plt.cm.get_cmap('Greys')


def _fetch_dir(directory, num=1, use_https=False):
    plt.ion()
    plt.show()
    for _ in range(num):
        img = captcha_source.fetch_image(use_https)
        plt.clf()
        plt.axis('off')
        plt.imshow(img)
        # http://stackoverflow.com/questions/12670101/matplotlib-ion-function
        # -fails-to-be-interactive
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
    c.clear_dir(c.training_set_dir)


def clear_training_chars():
    c.clear_dir(c.training_char_dir)


def clear_test_set():
    c.clear_dir(c.test_set_dir)


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
        print(
            'Requesting more images than stored, returning all available '
            'images')
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


def convert_train_image_to_char():
    total = 0
    success = 0
    cap = caprec.CaptchaRecognizer()
    for i in _list_png(c.training_set_dir):
        total += 1
        img = get_training_image(i[0:5])
        img_01 = cap.remove_noise_with_hsv(img)
        img_02 = cap.remove_noise_with_neighbors(img_01)
        img_02 = cap.remove_noise_with_neighbors(img_02)
        _, cut_line = cap.find_vertical_separation_line(img_02)
        img_list = cap.cut_images(img_02, cut_line)
        if len(img_list) == 5:
            success += 1
            print("Successfully converted {0} images out of {1} images".format(
                success, total))
            for j in range(5):
                mpimg.imsave(c.char_path(i[j], i), img_list[j], cmap=_cm_greys)

if __name__ == '__main__':
    pass
