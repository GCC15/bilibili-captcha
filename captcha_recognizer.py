# Handle image processing before giving over to captcha learner

import config as c
import captcha_source
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import helper

# Color map for grayscale images
_cm_greys = plt.cm.get_cmap('Greys')


class CaptchaRecognizer:
    def __init__(self, h_tol=6 / 360, s_tol=36 / 100, v_tol=40 / 100):
        self.h_tolerance = h_tol
        self.s_tolerance = s_tol
        self.v_tolerance = v_tol
        self.character_num = captcha_source.captcha_length
        self.char_width_min = 5
        self.char_width_max = 30
        self.char_height_min = 10
        self.char_height_max = 30

    # Try to partition a CAPTCHA into each char image
    # save_intermediate: whether I should save intermediate images
    def partition(self, img, save_intermediate=False, verbose=False):
        if save_intermediate:
            mpimg.imsave(c.temp_path('00.origin.png'), img)

        # 1
        img_01 = helper.time_func(
            'remove_noise_with_hsv' if verbose else None,
            lambda: self.remove_noise_with_hsv(img)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm_greys)

        # 2
        img_02 = helper.time_func(
            'remove_noise_with_neighbors' if verbose else None,
            lambda: helper.repeat(self.remove_noise_with_neighbors, 2)(img_01)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=_cm_greys)

        # 3
        labels, object_slices = helper.time_func(
            'segment_with_label' if verbose else None,
            lambda: self.segment_with_label(img_02)
        )
        if verbose:
            print('{} connected components found'.format(len(object_slices)))
        if save_intermediate:
            mpimg.imsave(c.temp_path('03.00000.png'), labels)
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
                            char_images[i], cmap=_cm_greys)
                return char_images
        if verbose:
            print('Warning: partition failed!')
        return None

    def recognize(self, img, save_intermediate=False, verbose=False):
        char_images = self.partition(img, save_intermediate, verbose)

        # TODO: hand over to the neural network
        # Should return a tuple, the first one is a boolean variable
        # indicating whether the recognition is successful. If so, the second
        # return value is the captch as a string
        return

    # Convert to a grayscale image using HSV
    def remove_noise_with_hsv(self, img):
        # Use number of occurrences to find the standard h, s, v
        # Convert to int so we can sort the colors
        # noinspection PyTypeChecker
        img_int = np.dot(np.rint(img * 255), np.power(256, np.arange(3)))
        color_array = helper.sort_by_occurrence(img_int.flatten())
        # 2nd most frequent
        std_color = color_array[1]
        std_b, mod = divmod(std_color, 256 ** 2)
        std_g, std_r = divmod(mod, 256)
        # print([std_r, std_g, std_b])
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
