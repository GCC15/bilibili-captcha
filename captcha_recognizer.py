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
from scipy import ndimage


# import skimage.morphology as morph
# import skimage.segmentation as seg


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


def _rgb_to_int(rgb):
    return int(colors.rgb2hex(rgb)[1:], 16)


def _int_to_rgb(n):
    return colors.hex2color('#{:06x}'.format(n))


# Color map for grayscale images
_cm_greys = plt.cm.get_cmap('Greys')


# Show image in matplotlib window
def _show_image(img, cmap=_cm_greys, title=None):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


# Resize characters to another shape
def _resize_image_to_standard(img, width, height):
    return sp.misc.imresize(img, (height, width))


class CaptchaRecognizer:
    def __init__(self):
        # TODO: tune the tolerance values
        self.h_tolerance = 6 / 360
        self.s_tolerance = 34 / 100
        self.v_tolerance = 60 / 100
        self.sep_constant = 0.03  # this means all in one column must be
        # white, if set to 0.04, bad for 'QN4EL'
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
        t0 = time.time()
        img_01 = self.remove_noise_with_hsv(img)
        t1 = time.time()
        if verbose:
            print('Time for remove_noise_with_hsv: {}'.format(t1 - t0))
        if save_intermediate:
            mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm_greys)

        # 2
        t0 = time.time()
        img_02 = self.remove_noise_with_neighbors(img_01)
        img_02 = self.remove_noise_with_neighbors(img_02)
        t1 = time.time()
        if verbose:
            print('Time for remove_noise_with_neighbors: {}'.format(t1 - t0))
        if save_intermediate:
            mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=_cm_greys)

        # 3
        t0 = time.time()
        labels, object_slices = self.segment_with_label(img_02)
        t1 = time.time()
        if save_intermediate:
            mpimg.imsave(c.temp_path('03.00000.png'), labels)
        if verbose:
            print('Time for segment: {}'.format(t1 - t0))
            print('{} connected components found'.format(len(object_slices)))
        # Arrange the segments from left to right
        xmin_arr = np.array(
            [object_slice[1].start for object_slice in object_slices]
        )
        char_images = [img_02[object_slices[i]] for i in xmin_arr.argsort()]

        # Check if segmentation was successful
        if len(char_images) == self.character_num:
            shapes = np.array(list(map(np.shape, char_images)))
            heights, widths = shapes[:, 0], shapes[:, 1]
            if verbose:
                print('Heights {}'.format(heights))
                print('Widths {}'.format(widths))
            if (np.all(heights >= self.char_height_min) and
                    np.all(heights <= self.char_height_max) and
                    np.all(widths >= self.char_width_min) and
                    np.all(widths <= self.char_width_max)):
                def resize(char_image):
                    return _resize_image_to_standard(
                        char_image,
                        self.char_width_std,
                        self.char_height_std
                    )

                char_images = list(map(resize, char_images))
                if save_intermediate:
                    for i in range(len(char_images)):
                        mpimg.imsave(
                            c.temp_path('03.char.{}.png'.format(i + 1)),
                            char_images[i], cmap=_cm_greys)
                return char_images
        if verbose:
            print('Warning: partition failed')
        return None

        # t0 = time.time()
        # img_02a = self.anneal(img_02)
        # t1 = time.time()
        # print('Annealing time: {}'.format(t1 - t0))
        # mpimg.imsave(c.temp_path('02a.anneal.png'), img_02a)

        # 3
        # img_03f = self.cut_images_by_floodfill(img_02)

        # img_03v, cut_line = self.find_vertical_separation_line(img_02)
        # mpimg.imsave(c.temp_path('03v.separate.png'), img_03v, cmap=_cm_greys)

        # 4
        # for i in range(len(image_cut)):
        #     mpimg.imsave(c.temp_path('04.cut{0}.png'.format(i + 1)),
        #                  image_cut[i], cmap=_cm_greys)
        # print(self.get_degree_of_similarity(image_cut[0], image_cut[1]))

    def recognize(self, img, save_intermediate=False, verbose=False):
        char_images = self.partition(
            img,
            save_intermediate=save_intermediate,
            verbose=verbose
        )

        # TODO: hand over to the neural network

        return

    # Convert to a grayscale image using HSV
    def remove_noise_with_hsv(self, img):
        # Use number of occurrences to find the standard h, s, v
        # Convert to int so we can sort the colors
        # t0 = time.time()
        # TODO: this line is too slow! optimize
        img_int = np.apply_along_axis(_rgb_to_int, 2, img)
        # t1 = time.time()
        # print('_rgb_to_int: {}'.format(t1 - t0))
        color_array = _sort_by_occurrence(img_int.flatten())
        # 2nd most frequent
        std_color = color_array[1]
        std_h, std_s, std_v = colors.rgb_to_hsv(_int_to_rgb(std_color))
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
    # TODO: optimize using vectorized operations
    def remove_noise_with_neighbors(self, img, neighbor_low=0, neighbor_high=7):
        height, width = img.shape
        new_img = img.copy()
        for y in range(height):
            for x in range(width):
                num_a = 0
                num_b = 0
                sum_color = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        y_neighbor = y + dy
                        if y_neighbor < 0 or y_neighbor >= height:
                            continue
                        x_neighbor = x + dx
                        if x_neighbor < 0 or x_neighbor >= width:
                            continue
                        color = img[y_neighbor, x_neighbor]
                        sum_color += color
                        if color == 1:
                            num_a += 1
                        elif color > 0:
                            num_b += 1
                if img[y, x] * 2 > sum_color:
                    new_img[y, x] = 0
                elif img[y, x] > 0:
                    if num_a <= neighbor_low:
                        new_img[y, x] = 0
                elif num_a + num_b >= neighbor_high:
                    new_img[y, x] = sum_color / 8
        return new_img

    def segment_with_label(self, img):
        # Next-nearest neighbors
        struct_nnn = np.ones((3, 3), dtype=int)
        labels, num_labels = ndimage.label(img > 0, structure=struct_nnn)
        # np.savetxt(c.temp_path('labels.txt'), labels, fmt='%d')
        object_slices = ndimage.find_objects(labels)
        return labels, object_slices

    def find_vertical_separation_line(self, img):
        sep_line_list = []
        sep_line_list_final = []
        new_img = img.copy()
        height, width = img.shape
        for x in range(width):
            if np.count_nonzero(img[:, x]) / height < self.sep_constant:
                sep_line_list.append(x)
                new_img[:, x] = 0.5
        for i in range(len(sep_line_list)):
            if i == 0 or sep_line_list[i] == (width - 1) or sep_line_list[i] - \
                    sep_line_list[i - 1] != 1:
                if i != 0 and sep_line_list[i] != (width - 1):
                    sep_line_list_final.append(sep_line_list[i - 1])
                sep_line_list_final.append(sep_line_list[i])
        return [new_img, sep_line_list_final]

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
