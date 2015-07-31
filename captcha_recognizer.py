from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# TODO


class CaptchaRecognizer:
    def recognize(self, img):
        plt.clf()
        plt.imshow(img)
        plt.show() # why it and imshow has to be used together? Also why the opened plot has to be closed before proceeding
        plt.hist(colors.rgb_to_hsv(img)[:, :, 0].flatten(), bins=512, range=(0, 1))
        plt.savefig('temp/00.origin.hue.hist.png')
