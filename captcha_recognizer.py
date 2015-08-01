from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import numpy as np
import config as c


class CaptchaRecognizer:
    def __init__(self):
        self.color_0 = 255
        self.color_1 = 0
        self.hue_tolerance = 5
        self.neighbor_low = 1
        self.neighbor_high = 5
        self.sep_constant = 0.016

    def recognize(self, img):
        # plt.clf()
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # img = img.convert("RGB")
        # plt.hist(colors.rgb_to_hsv(img)[:, :, 0].flatten(), bins=512, range=(0, 1))
        # plt.savefig('temp/00.origin.hue.hist.png')
        img.save(c.temp_path('00.origin.png'))
        img_hue = self.remove_noise_with_hue(img)
        img_hue.save(c.temp_path('01.hue.png'))
        img_neighbor = self.remove_noise_with_neighbors(img_hue)
        img_neighbor.save(c.temp_path('02.neighbor.png'))
        img_neighbor.show()
        self.find_vertical_sepration_line(img_neighbor)

    def remove_noise_with_hue(self, img):
        # img_rgb = img.convert('RGB')
        img_hsv = img.convert('HSV')
        img = img.convert('P')
        length, width = img.size
        new_img = Image.new('P', img.size, self.color_0)
        hist = img.histogram()
        sort_index = np.argsort(np.array(hist))[::-1]
        # stand_r = 0
        # stand_g = 0
        # stand_b = 0
        std_h = 0
        std_s = 0
        std_v = 0
        for x in range(length):
            for y in range(width):
                if img.getpixel((x, y)) == sort_index[1]:
                    # stand_r, stand_g, stand_b = img_rgb.getpixel((x, y))
                    std_h, std_s, std_v = img_hsv.getpixel((x, y))
                    break
        for x in range(length):
            for y in range(width):
                # r, g, b = img_rgb.getpixel((x, y))
                h, s, v = img_hsv.getpixel((x, y))
                """
                if abs(r - stand_r) <= 40 and abs(g - stand_g) <= 40 and abs(b - stand_b) <= 40:
                    new_img.putpixel((x, y), 0)
                """
                if abs(h - std_h) <= self.hue_tolerance:
                    new_img.putpixel((x, y), self.color_1)
        return new_img

    def remove_noise_with_neighbors(self, img):
        length, width = img.size
        new_img = img.copy()
        for x in range(length):
            for y in range(width):
                num_neighbors = 0
                for dx in [-1, 0, 1]:
                    x_neighbor = x + dx
                    if x_neighbor < 0 or x_neighbor >= length: continue
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        y_neighbor = y + dy
                        if y_neighbor < 0 or y_neighbor >= width: continue
                        if img.getpixel((x_neighbor, y_neighbor)) == self.color_1:
                            num_neighbors += 1
                if img.getpixel((x, y)) == self.color_1:
                    if num_neighbors <= self.neighbor_low:
                        new_img.putpixel((x, y), self.color_0)
                else:
                    if num_neighbors >= self.neighbor_high:
                        new_img.putpixel((x, y), self.color_1)
        return new_img

    def find_vertical_sepration_line(self,img):
        sep_line_list = []
        new_img = img.copy()
        length, width = img.size
        for i in range(length):
            background_color_num = 0
            for j in range(width):
                if img.getpixel((i,j)) != self.color_0:
                    background_color_num += 1
            if background_color_num/(width*1.0) < self.sep_constant:
                sep_line_list.append(i)
                for j in range(width):
                    new_img.putpixel((i,j),150)
        new_img.show()
        return sep_line_list



