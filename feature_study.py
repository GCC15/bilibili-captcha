# Trying to improve char partitioning

import config as c
import dataset_manager
import captcha_recognizer
from captcha_recognizer import _time_func, _repeat, _cm_greys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.morphology as morph
# import skimage.segmentation as seg


def main():
    c.clear_temp()

    img = dataset_manager.get_training_image()
    recognizer = captcha_recognizer.CaptchaRecognizer()
    mpimg.imsave(c.temp_path('00.origin.png'), img)

    # 1
    img_01 = _time_func(
        'remove_noise_with_hsv',
        lambda: recognizer.remove_noise_with_hsv(img)
    )
    mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm_greys)

    # 2
    img_02 = _time_func(
        'remove_noise_with_neighbors',
        lambda: _repeat(recognizer.remove_noise_with_neighbors, 2)(img_01)
    )
    mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=_cm_greys)

    img_03a = _time_func(
        'skeletonize',
        lambda: morph.skeletonize(img_02)
    )
    mpimg.imsave(c.temp_path('03a.skeleton.png'), img_03a, cmap=_cm_greys)

    # medial_axis is slow and gives inferior results.
    # img_03b = _time_func(
    #     'medial_axis',
    #     lambda: morph.medial_axis(img_02)
    # )
    # mpimg.imsave(c.temp_path('03b.medial_axis.png'), img_03b, cmap=_cm_greys)

    # img_04 = recognizer.anneal(img_03a)
    # mpimg.imsave(c.temp_path('04.anneal.png'), img_04)

if __name__ == '__main__':
    main()
