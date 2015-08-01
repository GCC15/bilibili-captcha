from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


# TODO


class CaptchaRecognizer:
    def recognize(self, img):
        # plt.clf()
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # img = img.convert("RGB")
        # plt.hist(colors.rgb_to_hsv(img)[:, :, 0].flatten(), bins=512, range=(0, 1))
        img.show()
        new_img = self.remove_noise(img)
        new_img.show()
        plt.savefig('temp/00.origin.hue.hist.png')

    def remove_noise(self, img):
        img_rgb = img.convert('RGB')
        img_hsv = img.convert('HSV')
        img = img.convert('P')
        length = img.size[0]
        width = img.size[1]
        new_img = Image.new('P', (length, width), 255)
        hist = img.histogram()
        sort_index = np.argsort(np.array(hist))[::-1]
        stand_r = 0
        stand_g = 0
        stand_b = 0
        stand_h = 0
        stand_s = 0
        stand_v = 0
        for x in range(length):
            for y in range(width):
                if img.getpixel((x, y)) == sort_index[1]:
                    stand_r, stand_g, stand_b = img_rgb.getpixel((x, y))
                    stand_h, stand_s, stand_v = img_hsv.getpixel((x, y))
                    break
        for x in range(length):
            for y in range(width):
                r, g, b = img_rgb.getpixel((x, y))
                h, s, v = img_hsv.getpixel((x, y))
                """
                if abs(r - stand_r) <= 40 and abs(g - stand_g) <= 40 and abs(b - stand_b) <= 40:
                    new_img.putpixel((x, y), 0)
                """
                if abs(h - stand_h) <=5:
                    new_img.putpixel((x, y), 0)
        return new_img
