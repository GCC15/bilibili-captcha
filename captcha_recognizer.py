# from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import numpy as np
import config as c


# https://en.wikipedia.org/wiki/Moore_neighborhood
def _chebyshev_neighbors(r=1):
    d = range(-r, r + 1, 1)
    neighbors = []
    for dy in d:
        for dx in d:
            if dy == 0 and dx == 0: continue
            neighbors.append((dy, dx))
    return neighbors


# https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
def _manhattan_neighbors(r=1):
    pass


# E.g. _sort_by_occurrence(np.array([1, 3, 3, 1, 2, 2, 2, 3, 4, 2]))
# Return: array([2, 3, 1, 4])
def _sort_by_occurrence(arr):
    arr = np.sort(arr)
    val_list = [-1]
    occur_list = [0]
    for val in arr:
        if val == val_list[-1]:
            occur_list[-1] += 1
        else:
            val_list.append(val)
            occur_list.append(1)
    occur_array = np.array(occur_list)
    sort_index = occur_array.argsort()[:0:-1]
    val_array = np.array(val_list)
    return val_array[sort_index]


def _rgb_to_int(rgb):
    return int(colors.rgb2hex(rgb)[1:], 16)


def _int_to_rgb(n):
    return colors.hex2color('#{:06x}'.format(n))


# Color map for binary images
_cm = plt.cm.get_cmap('Greys')


# Show binary image in matplotlib window
def _show_image(img):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=_cm)
    plt.show()


class CaptchaRecognizer:
    def __init__(self):
        self.h_tolerance = 5 / 360
        self.s_tolerance = 30 / 100
        self.v_tolerance = 40 / 100
        self.neighbor_low = 0
        self.neighbor_high = 5
        self.sep_constant = 0.016

    def recognize(self, img):
        mpimg.imsave(c.temp_path('00.origin.png'), img)

        img_01 = self.remove_noise_with_hsv(img)
        mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm)

        img_02 = self.remove_noise_with_neighbors(img_01, _chebyshev_neighbors(1), 1, 5)
        mpimg.imsave(c.temp_path('02.neighbor.1.1.5.png'), img_02, cmap=_cm)

        img_02 = self.remove_noise_with_neighbors(img_01, _chebyshev_neighbors(1), 1, 6)
        mpimg.imsave(c.temp_path('02.neighbor.1.1.6.png'), img_02, cmap=_cm)

        img_02 = self.remove_noise_with_neighbors(img_01, _chebyshev_neighbors(2), 3, 14)
        mpimg.imsave(c.temp_path('02.neighbor.2.3.14.png'), img_02, cmap=_cm)

        img_03 = self.find_vertical_separation_line(img_02)
        mpimg.imsave(c.temp_path('03.separate.png'), img_03, cmap=_cm)

    def remove_noise_with_hsv(self, img):
        # Use number of occurrences to find the standard h, s, v
        # Convert to int so we can sort the colors
        img_int = np.apply_along_axis(_rgb_to_int, 2, img)
        color_array = _sort_by_occurrence(img_int.flatten())
        std_color = color_array[1]
        std_h, std_s, std_v = colors.rgb_to_hsv(_int_to_rgb(std_color))
        # print(std_h * 360, std_s * 100, std_v * 100)
        Y, X, _ = img.shape
        new_img = np.zeros((Y, X))
        img_hsv = colors.rgb_to_hsv(img)
        for y in range(Y):
            for x in range(X):
                h, s, v = img_hsv[y, x, :]
                if abs(h - std_h) <= self.h_tolerance and \
                                abs(s - std_s) <= self.s_tolerance and \
                                abs(v - std_v) <= self.v_tolerance:
                    new_img[y, x] = 1
        return new_img

    def remove_noise_with_neighbors(self, img, neighbors, neighbor_low, neighbor_high):
        Y, X = img.shape
        new_img = img.copy()
        for y in range(Y):
            for x in range(X):
                num_neighbors = 0
                for dy, dx in neighbors:
                    y_neighbor = y + dy
                    if y_neighbor < 0 or y_neighbor >= Y: continue
                    x_neighbor = x + dx
                    if x_neighbor < 0 or x_neighbor >= X: continue
                    if img[y_neighbor, x_neighbor]:
                        num_neighbors += 1
                if img[y, x]:
                    if num_neighbors <= neighbor_low:
                        new_img[y, x] = 0
                else:
                    if num_neighbors >= neighbor_high:
                        new_img[y, x] = 1
        return new_img

    def find_vertical_separation_line(self, img):
        sep_line_list = []
        new_img = img.copy()
        Y, X = img.shape
        for x in range(X):
            if np.count_nonzero(img[:, x]) / Y < self.sep_constant:
                sep_line_list.append(x)
                new_img[:, x] = 0.5
        return new_img
