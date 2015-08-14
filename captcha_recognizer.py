# Handle image processing before handing over to captcha learner

import matplotlib.colors as colors
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage

import config as c
from helper import time_func, cm_greys, repeat, sort_by_occurrence
from captcha_provider import BilibiliCaptchaProvider
import captcha_learn


class CaptchaRecognizer:
    def __init__(self, captcha_provider=BilibiliCaptchaProvider(),
                 h_tol=6 / 360,
                 s_tol=36 / 100,
                 v_tol=40 / 100):
        # Three parameters to be used in remove_noise_with_hsv
        self.h_tolerance = h_tol
        self.s_tolerance = s_tol
        self.v_tolerance = v_tol

        self.character_num = captcha_provider.seq_length

        # Four parameters to be used in partition
        self.char_width_min = 5
        self.char_width_max = 30
        self.char_height_min = 10
        self.char_height_max = 30

    # Try to partition a CAPTCHA into each char image
    # save_intermediate: whether I should save intermediate images
    # TODO: Forced partition
    def partition(self, img, save_intermediate=False, verbose=False):
        if save_intermediate:
            mpimg.imsave(c.temp_path('00.origin.png'), img)

        # step 1
        img_01 = time_func(
            'remove_noise_with_hsv' if verbose else None,
            lambda: self.remove_noise_with_hsv(img)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=cm_greys)

        # step 2
        img_02 = time_func(
            'remove_noise_with_neighbors' if verbose else None,
            lambda: repeat(self.remove_noise_with_neighbors, 2)(img_01)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=cm_greys)

        # step 3
        labels, object_slices = time_func(
            'segment_with_label' if verbose else None,
            lambda: self.segment_with_label(img_02)
        )
        if verbose:
            print('{} connected components found'.format(len(object_slices)))
        if save_intermediate:
            mpimg.imsave(c.temp_path('03.00000.png'), labels)

        # step 4

        # Arrange the segments from left to right
        xmin_arr = np.array([s[1].start for s in object_slices])
        sort_index = xmin_arr.argsort()
        char_images = []
        # noinspection PyTypeChecker
        for i in sort_index:
            char_image = img_02.copy()
            char_image[labels != i + 1] = 0
            char_image = char_image[object_slices[i]]
            char_images.append(char_image)

        # Check if segmentation was successful
        if len(char_images) == self.character_num:
            shapes = np.array(list(map(np.shape, char_images)))
            heights, widths = shapes[:, 0], shapes[:, 1]
            if verbose:
                print('Heights {}'.format(heights))
                print('Widths {}'.format(widths))
            # noinspection PyTypeChecker
            if (np.all(heights >= self.char_height_min) and
                    np.all(heights <= self.char_height_max) and
                    np.all(widths >= self.char_width_min) and
                    np.all(widths <= self.char_width_max)):

                if save_intermediate:
                    for i in range(len(char_images)):
                        mpimg.imsave(
                            c.temp_path('03.char.{}.png'.format(i + 1)),
                            char_images[i], cmap=cm_greys)
                return char_images
        if verbose:
            print('Warning: partition failed!')
        return None

    # Recognize the captcha
    def recognize(self, img, save_intermediate=False, verbose=False,
                  reconstruct=False):
        seq = []
        char_images = self.partition(img, save_intermediate, verbose)
        if reconstruct:
            captcha_learn.reconstruct_model()
        if char_images is not None and len(char_images) == self.character_num:
            success = True

            def predict():
                nonlocal seq
                for i in range(len(char_images)):
                    seq.append(captcha_learn.predict(char_images[i]))

            time_func('predict' if verbose else None, predict)
            seq = ''.join(seq)
        else:
            success = False
        return success, seq

    # Convert to a grayscale image using HSV
    def remove_noise_with_hsv(self, img):
        # Use number of occurrences to find the standard h, s, v
        # Convert to int so we can sort the colors
        # noinspection PyTypeChecker
        img_int = np.dot(np.rint(img * 255), np.power(256, np.arange(3)))
        color_array = sort_by_occurrence(img_int.flatten())
        # standard color is the 2nd most frequent color
        std_color = color_array[1]
        std_b, mod = divmod(std_color, 256 ** 2)
        std_g, std_r = divmod(mod, 256)
        # noinspection PyTypeChecker
        std_h, std_s, std_v = colors.rgb_to_hsv(
            np.array([std_r, std_g, std_b]) / 255
        )
        # print(std_h * 360, std_s * 100, std_v * 100)
        height, width, _ = img.shape
        img_hsv = colors.rgb_to_hsv(img)
        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_mask = np.abs(h - std_h) > self.h_tolerance
        s_mask = np.abs(s - std_s) > self.s_tolerance
        delta_v = np.abs(v - std_v)
        v_mask = delta_v > self.v_tolerance
        hsv_mask = np.logical_or(
            np.logical_or(
                h_mask, s_mask
            ), v_mask
        )
        new_img = 1 - delta_v
        new_img[hsv_mask] = 0
        # Three types of grayscale colors in new_img:
        # Type A: 1. Outside noise, or inside point.
        # Type B: between 0 and 1. Outside noise, or contour point.
        # Type C: 0. Inside noise, or background.
        return new_img

    def remove_noise_with_neighbors(self, img, neighbor_low=0, neighbor_high=7):
        height, width = img.shape
        pad_shape = height + 2, width + 2
        img_pad_sum = np.zeros(pad_shape)
        img_pad_a = np.zeros(pad_shape)
        img_pad_b = np.zeros(pad_shape)
        neighbors = [-1, 0, 1]
        # Add padding in a vectorized manner
        for dy in neighbors:
            for dx in neighbors:
                if dy == 0 and dx == 0:
                    continue
                s = (slice(dy + 1, dy - 1 if dy - 1 else None),
                     slice(dx + 1, dx - 1 if dx - 1 else None))
                img_pad_sum[s] += img
                img_pad_a[s] += img == 1
                img_pad_b[s] += np.logical_and(img > 0, img < 1)
        # Remove padding
        s = [slice(1, -1)] * 2
        img_pad_sum = img_pad_sum[s]
        img_pad_a = img_pad_a[s]
        img_pad_b = img_pad_b[s]
        new_img = img.copy()
        mask = np.logical_and(img == 0, img_pad_a + img_pad_b >= neighbor_high)
        new_img[mask] = img_pad_sum[mask] / 8
        new_img[img * 1.3 > img_pad_sum] = 0
        new_img[img_pad_a <= neighbor_low] = 0
        return new_img

    def segment_with_label(self, img):
        # Next-nearest neighbors
        struct_nnn = np.ones((3, 3), dtype=int)
        labels, _ = ndimage.label(img, structure=struct_nnn)
        # np.savetxt(c.temp_path('labels.txt'), labels, fmt='%d')
        object_slices = ndimage.find_objects(labels)
        return labels, object_slices
