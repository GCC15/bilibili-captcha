# A source from which we fetch captcha images

import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import helper

captcha_length = 5
#       'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')
charset = set(chars)
session = None


def _get_captcha_url():
    return 'https://www.bilibili.com/captcha'


def fetch_captcha(retry_limit=3, url=_get_captcha_url()):
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


def fill_captcha(url=_get_captcha_url()):
    image = fetch_captcha(3, url)
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
    # success, string = captcha_recognizer.CaptchaRecognizer().recognize(image)
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
