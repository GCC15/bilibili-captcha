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

    def remove_noise(img):
        img = img.convert("P")
        length = img.size[0]
        width = img.size[1]
        new_img = Image.new("P", (length, width), 255)
        hist = img.histogram()
        sort_index = np.argsort(np.array(hist))[::-1]
        for x in range(length):
            for y in range(width):
                if img.getpixel((x, y)) == sort_index[1]:
                    new_img.putpixel((x, y), 0)
        return new_img
