# from PIL import Image
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import numpy as np
import numpy.linalg as la
import config as c


# https://en.wikipedia.org/wiki/Lennard-Jones_potential
def _LJ(r, delta=3):
    ret = np.power(delta / r, 12) - 2 * np.power(delta / r, 6)
    return ret


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


# Show grayscale image in matplotlib window
def _show_image(img, cmap=_cm_greys, title=None):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


class CaptchaRecognizer:
    def __init__(self):
        self.length = 0
        self.width = 0
        self.h_tolerance = 6 / 360
        self.s_tolerance = 34 / 100
        self.v_tolerance = 60 / 100
        self.neighbor_low = 0
        self.neighbor_high = 5
        self.sep_constant = 0.03  # this means all in one column must be white, if set to 0.04, bad for 'QN4EL'
        self.character_num = 5

    def recognize(self, img):
        width, length, _ = img.shape
        self.width = width
        self.length = length
        mpimg.imsave(c.temp_path('00.origin.png'), img)

        # 1
        img_01 = self.remove_noise_with_hsv(img)
        mpimg.imsave(c.temp_path('01.hsv.png'), img_01, cmap=_cm_greys)

        # 2
        img_02 = self.remove_noise_with_neighbors(img_01)
        img_02 = self.remove_noise_with_neighbors(img_02)
        mpimg.imsave(c.temp_path('02.neighbor.png'), img_02, cmap=_cm_greys)

        t0 = time.time()
        img_02a = self.anneal(img_02)
        print(time.time() - t0)
        mpimg.imsave(c.temp_path('02a.anneal.png'), img_02a)

        return

        # 3
        img_03, cut_line = self.find_vertical_separation_line(img_02)
        mpimg.imsave(c.temp_path('03.separate.png'), img_03, cmap=_cm_greys)

        # 4
        image_cut = self.cut_images(img_02, cut_line)
        print(self.get_degree_of_similarity(image_cut[0], image_cut[1]))
        return

    # Convert to a grayscale image using HSV
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
                if (abs(h - std_h) <= self.h_tolerance and
                            abs(s - std_s) <= self.s_tolerance and
                            abs(v - std_v) <= self.v_tolerance):
                    delta_v = abs(v - std_v)
                    if delta_v <= 1e-4:
                        new_img[y, x] = 1
                    else:
                        new_img[y, x] = 1 - delta_v
        # Three types of grayscale colors in new_img:
        # Type A: 1. Outside noise, or inside point.
        # Type B: between 0 and 1. Outside noise, or contour point.
        # Type C: 0. Inside noise, or background.
        return new_img

    def remove_noise_with_neighbors(self, img, neighbor_low=0, neighbor_high=7):
        Y, X = img.shape
        new_img = img.copy()
        for y in range(Y):
            for x in range(X):
                num_a = 0
                num_b = 0
                sum_color = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        y_neighbor = y + dy
                        if y_neighbor < 0 or y_neighbor >= Y:
                            continue
                        x_neighbor = x + dx
                        if x_neighbor < 0 or x_neighbor >= X:
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

    def find_vertical_separation_line(self, img):
        sep_line_list = []
        sep_line_list_final = []
        new_img = img.copy()
        Y, X = img.shape
        for x in range(X):
            if np.count_nonzero(img[:, x]) / Y < self.sep_constant:
                sep_line_list.append(x)
                new_img[:, x] = 0.5
        for i in range(len(sep_line_list)):
            if i == 0 or sep_line_list[i] == (X - 1) or sep_line_list[i] - \
                    sep_line_list[i - 1] != 1:
                if i != 0 and sep_line_list[i] != (X - 1):
                    sep_line_list_final.append(sep_line_list[i - 1])
                sep_line_list_final.append(sep_line_list[i])
        return [new_img, sep_line_list_final]

    def cut_images(self, img, cut_line):
        cut_image_list = []
        resized_image_list = []
        print(cut_line)
        if len(cut_line) > 2 * (self.character_num + 1):
            print("Abnormal, the image will be cut into more than 5 pieces")
        if cut_line[0] == 0:
            for i in range(int(len(cut_line) / 2) - 1):
                cut_image_list.append(
                    img[:, cut_line[2 * i + 1]:cut_line[2 * i + 2]])
                mpimg.imsave(c.temp_path('cut{0}.png'.format(i + 1)),
                             img[:, cut_line[2 * i + 1]:cut_line[2 * i + 2]],
                             cmap=_cm_greys)
        else:
            for i in range(int(len(cut_line) / 2)):
                if i == 0:
                    cut_image_list.append(img[:, 0:cut_line[0]])
                    mpimg.imsave(c.temp_path('04.cut{0}.png'.format(i + 1)),
                                 img[:, 0:cut_line[0]], cmap=_cm_greys)
                else:
                    cut_image_list.append(
                        img[:, cut_line[2 * i - 1]:cut_line[2 * i]])
                    mpimg.imsave(c.temp_path('04.cut{0}.png'.format(i + 1)),
                                 img[:, cut_line[2 * i - 1]:cut_line[2 * i]],
                                 cmap=_cm_greys)
        for image in cut_image_list:
            resized_image_list.append(self.resize_image_to_standard(image))
        return resized_image_list

    # Requires two images to be of the same size and both black / white
    def get_degree_of_similarity(self, img1, img2):
        width1, length1 = img1.shape
        width2, length2 = img2.shape
        if width1 != width2 or length1 != length2:
            raise ValueError("Two images of different size are compared")
        point_num = 0
        for x in range(length1):
            for y in range(width1):
                if img1[y, x].all() == img2[y, x].all():
                    point_num += 1
        return point_num / (width1 * length1 * 1.0)

    def resize_image_to_standard(self, img):
        if self.width is None or self.length is None:
            raise ValueError("Standard size unknown")
        width, length = img.shape
        if self.width != width:
            raise ValueError("The width of the image is not standard")
        img_resized = img.resize((int(self.width), round(self.length/(self.character_num*1.0) - 2)))  # TODO: bug
        return img_resized

    # https://en.wikipedia.org/wiki/Simulated_annealing
    def anneal(self, img, num_steps=500):
        Y, X = img.shape
        # TODO: User RGB for now, just for visualization
        new_img = np.zeros((Y, X, 3))
        for i in range(3):
            new_img[:, :, i] = 1 - img.copy()
        positions = []
        for y in range(Y):
            for x in range(X):
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
            beta = 10 + step / 200
            # Choose a position randomly, and invert the state
            p = np.random.randint(num_positions)
            y, x = positions[p]
            delta_E = np.nansum(_LJ(la.norm(positions[particles] - positions[p], axis=1)))
            if particles[p]:
                delta_E = -delta_E
            if delta_E < 0:
                accept = True
            else:
                accept = (np.random.rand() < np.exp(-beta * delta_E))
            if accept:
                E += delta_E
                particles[p] = not particles[p]
                new_img[y, x, 0] = particles[p]
            if step % 50 == 0:
                print('Step {}. beta {}. E {}'.format(step, beta, E))
                # _show_image(new_img, title=step)
                # plt.pause(0.1)
        plt.ioff()
        return new_img
