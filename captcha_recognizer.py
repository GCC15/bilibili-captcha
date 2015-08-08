# Handle image processing before giving over to captcha learner

import config as c
import captcha_source
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.misc
# import skimage.morphology as morph
# import skimage.segmentation as seg
from scipy import ndimage


# A generic function timer
def _time_func(tag, func):
    t0 = time.time() if tag else None
    ret = func()
    if tag:
        t1 = time.time()
        print('Time for {}: {}'.format(tag, t1 - t0))
    return ret


# Compose a single-argument function n times
def _repeat(func, n):
    def ret(x):
        for i in range(n):
            x = func(x)
        return x

    return ret


# https://en.wikipedia.org/wiki/Lennard-Jones_potential
def _lj(r, delta=4):
    return np.power(delta / r, 12) - 2 * np.power(delta / r, 6)


# # https://en.wikipedia.org/wiki/Moore_neighborhood
# def _chebyshev_neighbors(r=1):
#     d = range(-r, r + 1)
#     neighbors = []
#     for dy in d:
#         for dx in d:
#             if dy == 0 and dx == 0:
#                 continue
#             neighbors.append((dy, dx))
#     return neighbors
#
#
# # https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
# def _manhattan_neighbors(r=1):
#     neighbors = []
#     for dy in range(-r, r + 1):
#         xx = r - abs(dy)
#         for dx in range(-xx, xx + 1):
#             if dy == 0 and dx == 0:
#                 continue
#             neighbors.append((dy, dx))
#     return neighbors


# E.g. _sort_by_occurrence(np.array([1, 3, 3, 1, 2, 2, 2, 3, 4, 2]))
# Return: array([2, 3, 1, 4])
def _sort_by_occurrence(arr):
    u, counts = np.unique(arr, return_counts=True)
    sort_index = counts.argsort()[::-1]
    return u[sort_index]


# Color map for grayscale images
_cm_greys = plt.cm.get_cmap('Greys')


# Show image in matplotlib window
def _show_image(img, cmap=_cm_greys, title=None):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


class CaptchaRecognizer:
    def __init__(self, h_tol=6 / 360, s_tol=36 / 100, v_tol=40 / 100):
        self.h_tolerance = h_tol
        self.s_tolerance = s_tol
        self.v_tolerance = v_tol
        self.character_num = captcha_source.captcha_length
        # TODO: adjust the values below
        self.char_width_std = 15
        self.char_height_std = 30
        self.char_width_min = 5
        self.char_width_max = 30
        self.char_height_min = 15
        self.char_height_max = 30

    # Try to partition a CAPTCHA into each char image
    # save_intermediate: whether I should save intermediate images
    def partition(self, img, save_intermediate=False, verbose=False):
        if save_intermediate:
            mpimg.imsave(c.temp_path('00.origin.png'), img)

        # 1
        img_01 = _time_func(
            'remove_noise_with_hsv' if verbose else None,
            lambda: self.remove_noise_with_hsv(img)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm_greys)

        # 2
        img_02 = _time_func(
            'remove_noise_with_neighbors' if verbose else None,
            lambda: _repeat(self.remove_noise_with_neighbors, 2)(img_01)
        )
        if save_intermediate:
            mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=_cm_greys)

        # 3
        labels, object_slices = _time_func(
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
                # def resize(char_image):
                #     return sp.misc.imresize(
                #         char_image,
                #         (self.char_height_std, self.char_width_std)
                #     )
                # char_images = list(map(resize, char_images))
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

        return

    # Convert to a grayscale image using HSV
    def remove_noise_with_hsv(self, img):
        # Use number of occurrences to find the standard h, s, v
        # Convert to int so we can sort the colors
        # noinspection PyTypeChecker
        img_int = np.dot(np.rint(img * 255), np.power(256, np.arange(3)))
        color_array = _sort_by_occurrence(img_int.flatten())
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

    # TODO: how to improve this process?
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
                # print(s)
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
        new_img[img * 2 > img_pad_sum] = 0
        new_img[img_pad_a <= neighbor_low] = 0
        return new_img

    def segment_with_label(self, img):
        # Next-nearest neighbors
        struct_nnn = np.ones((3, 3), dtype=int)
        labels, _ = ndimage.label(img, structure=struct_nnn)
        # np.savetxt(c.temp_path('labels.txt'), labels, fmt='%d')
        object_slices = ndimage.find_objects(labels)
        return labels, object_slices

    def cut_images_by_floodfill(self, img):
        height, width = img.shape
        aux_list = []
        region_point = []
        for x in range(width):
            for y in range(height):
                if img[y, x] == 1:
                    start = (y, x)
                    aux_list.append(start)
                    break
        while len(aux_list) != 0:
            (y, x) = aux_list.pop()
            if (y, x) not in region_point:
                region_point.append((y, x))
            if x + 1 < width and img[y, x + 1] == 1 and (
                    y, x + 1) not in region_point:
                aux_list.append((y, x + 1))
            if x - 1 >= 0 and img[y, x - 1] == 1 and (
                    y, x - 1) not in region_point:
                aux_list.append((y, x + 1))
            if y + 1 < height and img[y + 1, x] == 1 and (
                        y + 1, x) not in region_point:
                aux_list.append((y + 1, x))
            if y - 1 >= 0 and img[y - 1, x] == 1 and (
                        y - 1, x) not in region_point:
                aux_list.append((y - 1, x))
        print(region_point)

    # https://en.wikipedia.org/wiki/Simulated_annealing
    def anneal(self, img, num_steps=500):
        np.seterr(divide='ignore', invalid='ignore')
        height, width = img.shape
        # TODO: Use RGB for now, just for visualization
        new_img = np.zeros((height, width, 3))
        for i in range(3):
            new_img[:, :, i] = 1 - img.copy()
        positions = []
        for y in range(height):
            for x in range(width):
                if img[y, x] == 1:
                    new_img[y, x, 0] = 1
                    positions.append((y, x))
        positions = np.array(positions)
        num_positions = positions.shape[0]
        print('{} Positions'.format(num_positions))
        particles = np.ones(num_positions, dtype=bool)
        # plt.ion()
        # _show_image(new_img)
        # TODO: Just for testing
        E = 0
        # for p in range(num_positions):
        #     for q in range(p + 1, num_positions):
        #         E += _LJ(la.norm(positions[q] - positions[p]))
        for step in range(num_steps):
            beta = 10 + step / 50
            # Choose a position randomly, and invert the state
            p = np.random.randint(num_positions)
            y, x = positions[p]
            # noinspection PyTypeChecker
            delta_energy = np.nansum(
                _lj(la.norm(positions[particles] - positions[p], axis=1)))
            if particles[p]:
                delta_energy = -delta_energy
            if delta_energy < 0:
                accept = True
            else:
                accept = (random.random() < np.exp(-beta * delta_energy))
            if accept:
                E += delta_energy
                particles[p] = not particles[p]
                new_img[y, x, 0] = particles[p]
            if step % 50 == 0:
                print('Step {}. beta {}. E {}'.format(step, beta, E))
                # _show_image(new_img, title=step)
                # plt.pause(0.1)
        # plt.ioff()
        return new_img
