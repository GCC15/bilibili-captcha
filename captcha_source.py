# A source from which we fetch captcha images
# TODO: interactive test

import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import captcha_recognizer

captcha_length = 5
#       'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')
charset = set(chars)
session = None

def _get_captcha_url(use_https=False):
    return ('https' if use_https else 'http') + '://www.bilibili.com/captcha'


def canonicalize(seq):
    return seq.upper()


def fetch_image(use_https=False, retry_limit=3):
    global session
    session = requests.session()
    url = _get_captcha_url(use_https)
    print('Fetching CAPTCHA image from {}'.format(url))
    r = None
    for num_retries in range(retry_limit):
        if num_retries > 0:
            print('num_retries = {}'.format(num_retries))
        try:
            r = session.get(url)
            break
        except Exception as e:
            print(e)
    return None if r is None else mpimg.imread(BytesIO(r.content))


# TODO: Bugs, probably need header and cookie
def fill_captcha(use_https=False):
    image = fetch_image(use_https)
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
        if len(captcha) == captcha_length:
            break
    plt.ioff()
    #captcha = captcha_recognizer.CaptchaRecognizer().recognize(image)
    url = 'https' if use_https else 'http' + '://account.bilibili.com/register/mail'
    data = {'vd': captcha, 'action': 'checkVd'}
    r = session.post(url, data=data)
    print(r.json())
    if r.json()['status'] == 'false':
        return False
    elif r.json()['status'] == 'True':
        return True