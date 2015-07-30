import requests
from io import BytesIO
from numpy import *
from scipy import ndimage
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_captcha_url(use_https=False):
    return ('http' if use_https else 'https') + '://www.bilibili.com/captcha'


class CaptchaSource:
    def __init__(self, use_https=False):
        r = requests.get(get_captcha_url(use_https))
        img0 = mpimg.imread(BytesIO(r.content))
        print(img0.shape)
        # plt.axis('off')
        # plt.imshow(img)
        # plt.show()
        mpimg.imsave('temp/00.origin.00.png', img0)
        plt.clf()
        plt.hist(rgb_to_hsv(img0)[:, :, 0].flatten(), bins=512, range=(0, 1))
        plt.savefig('temp/00.origin.hue.hist.png')

if __name__ == '__main__':
    captcha_source = CaptchaSource(True)
