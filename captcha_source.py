# Class for sources from which we fetch captcha images
# This module is designed to be stand alone, independent of other modules in the
# project. The main focus of this project is to break the captcha of
# www.bilibili.com

import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

bilibili_captcha_length = 5
#                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
bilibili_chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')
bilibili_charset = set(bilibili_chars)
bilibili_url = 'https://www.bilibili.com/captcha'
session = None


# url is the url where we fetch captcha
# captcha_length is the number of characters in a captcha, must be a fixed
# number
# chars are the characters that may appear in the captcha
class CaptchaSource:
    def __init__(self, url=bilibili_url,
                 captcha_length=bilibili_captcha_length,
                 chars=bilibili_chars):
        self.url = url
        self.captcha_length = captcha_length
        self.chars = chars
        self.charset = set(chars)

    def fetch_captcha(self, retry_limit=3):
        url = self.url
        global session
        session = requests.session()
        header = {
            'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) "
                          "Gecko/20100101 Firefox/34.0",
        }
        print('Fetching CAPTCHA image from {}'.format(url))
        r = None
        for num_retries in range(retry_limit):
            if num_retries > 0:
                print('num_retries = {}'.format(num_retries))
            try:
                r = session.get(url, headers=header)
                break
            except Exception as e:
                print(e)
        return None if r is None else mpimg.imread(BytesIO(r.content))

    def fill_captcha(self):
        image = self.fetch_captcha(3)
        if session is None:
            raise ValueError('No session found')
        plt.ion()
        plt.show()
        plt.clf()
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        plt.pause(1e-2)
        while True:
            captcha = input('Enter the char sequence: ')
            if len(captcha) == self.captcha_length:
                break
        plt.ioff()
        # success, string = captcha_recognizer.CaptchaRecognizer().recognize(
        # image)
        # if success:
        #   captcha = string
        url = 'https://account.bilibili.com/register/mail'
        header = {
            'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) "
                          "Gecko/20100101 Firefox/34.0",
            'Host': "account.bilibili.com",
            'Referer': url,
        }
        data = {'vd': captcha, 'action': "checkVd"}
        r = session.post(url, headers=header, data=data)
        print(r.json())
        if r.json()['status'] == 'false':
            return False
        elif r.json()['status'] == 'True':
            return True
