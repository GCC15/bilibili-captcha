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
cookie = None


def _get_captcha_url(use_https=False):
    return ('https' if use_https else 'http') + '://www.bilibili.com/captcha'


def canonicalize(seq):
    return seq.upper()


def fetch_image(use_https=False, retry_limit=3):
    global session
    global cookie
    session = requests.session()
    url = _get_captcha_url(use_https)
    header = {
        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0",
    }
    print('Fetching CAPTCHA image from {}'.format(url))
    r = None
    for num_retries in range(retry_limit):
        if num_retries > 0:
            print('num_retries = {}'.format(num_retries))
        try:
            r = session.get(url, headers=header)
            # print(r.request.headers)
            # cookie = r.request.headers['cookie']
            break
        except Exception as e:
            print(e)
    return None if r is None else mpimg.imread(BytesIO(r.content))


# TODO: Bugs, probably need header and cookie
def fill_captcha(use_https=True):
    image = fetch_image(use_https)
    if session is None:# or cookie is None:
        raise ValueError('No session or session id found')
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
    url = ('https' if use_https else 'http') + '://account.bilibili.com/register/mail'
    header = {
        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0",
        'Host': "account.bilibili.com",
        'Referer': url,
        # 'Cookie': cookie,
        'Origin': ('https' if use_https else 'http') +  '://account.bilibili.com'
    }
    data = {'vd': captcha, 'action': "checkVd"}
    # print(cookie)
    r = session.post(url, headers=header, data=data)
    print(r.request.headers)
    print(data)
    print(r.json())
    if r.json()['status'] == 'false':
        return False
    elif r.json()['status'] == 'True':
        return True
