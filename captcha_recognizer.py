# from PIL import Image

# TODO

class CaptchaRecognizer:
    def recognize(self, img):
        plt.clf()
        plt.hist(rgb_to_hsv(img)[:, :, 0].flatten(), bins=512, range=(0, 1))
        plt.savefig('temp/00.origin.hue.hist.png')
