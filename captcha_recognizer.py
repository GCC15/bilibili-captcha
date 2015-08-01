# from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import numpy as np
import config as c

# Color map for binary images
cm = plt.cm.get_cmap('Greys')


# Show binary image in matplotlib window
def show_image(img):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=cm)
    plt.show()


class CaptchaRecognizer:
    def __init__(self):
        self.hue_tolerance = 0.02
        self.neighbor_low = 1
        self.neighbor_high = 5
        self.sep_constant = 0.016

    def recognize(self, img):
        mpimg.imsave(c.temp_path('00.origin.png'), img)

        img_hue = self.remove_noise_with_hue(img)
        mpimg.imsave(c.temp_path('01.hue.png'), img_hue, cmap=cm)

        img_neighbor = self.remove_noise_with_neighbors(img_hue)
        mpimg.imsave(c.temp_path('02.neighbor.png'), img_neighbor, cmap=cm)

        img_seperate = self.find_vertical_separation_line(img_neighbor)
        mpimg.imsave(c.temp_path('03.seperate.png'), img_seperate, cmap=cm)

    def remove_noise_with_hue(self, img):
        img_hsv = colors.rgb_to_hsv(img)
        hue_array = img_hsv[:, :, 0].flatten()
        hue_array.sort()
        hue_list = [-1]
        occur_list = [0]
        max_occurs = 1
        for hue in hue_array:
            if hue == hue_list[-1]:
                occur_list[-1] += 1
            else:
                if occur_list[-1] > max_occurs:
                    max_occurs = occur_list[-1]
                hue_list.append(hue)
                occur_list.append(1)
        occur_array = np.array(occur_list)
        sort_index = occur_array.argsort()[::-1]
        std_h = hue_list[sort_index[1]]
        Y, X, _ = img.shape
        new_img = np.zeros((Y, X))
        for y in range(Y):
            for x in range(X):
                h, s, v = img_hsv[y, x, :]
                if abs(h - std_h) <= self.hue_tolerance:
                    new_img[y, x] = 1
        return new_img

    def remove_noise_with_neighbors(self, img):
        Y, X = img.shape
        new_img = img.copy()
        for y in range(Y):
            for x in range(X):
                num_neighbors = 0
                for dy in [-1, 0, 1]:
                    y_neighbor = y + dy
                    if y_neighbor < 0 or y_neighbor >= Y: continue
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        x_neighbor = x + dx
                        if x_neighbor < 0 or x_neighbor >= X: continue
                        if img[y_neighbor, x_neighbor]:
                            num_neighbors += 1
                if img[y, x]:
                    if num_neighbors <= self.neighbor_low:
                        new_img[y, x] = 0
                else:
                    if num_neighbors >= self.neighbor_high:
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
