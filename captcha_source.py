import requests
from io import BytesIO
from numpy import *
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_captcha_url(use_https=False):
    return ('http' if use_https else 'https') + '://www.bilibili.com/captcha'


class CaptchaSource:
    def __init__(self, use_https=False):
        r = requests.get(get_captcha_url(use_https))
        img = mpimg.imread(BytesIO(r.content))
        print(img.shape)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        plt.hist(rgb_to_hsv(img)[:, :, 0].flatten(), bins=256, range=(0, 1))
        plt.show()


if __name__ == '__main__':
    captcha_source = CaptchaSource()
