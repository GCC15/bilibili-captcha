# A source from which we fetch captcha images

import requests
from io import BytesIO
import matplotlib.image as mpimg

captcha_length = 5
#       'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')
charset = set(chars)


def _get_captcha_url(use_https=False):
    return ('http' if use_https else 'https') + '://www.bilibili.com/captcha'


def canonicalize(seq):
    return seq.upper()


def fetch_image(use_https=False, retry_limit=3):
    url = _get_captcha_url(use_https)
    print('Fetching CAPTCHA image from {}'.format(url))
    r = None
    for num_retries in range(retry_limit):
        if num_retries > 0:
            print('num_retries = {}'.format(num_retries))
        try:
            r = requests.get(url)
            break
        except Exception as e:
            print(e)
    return None if r is None else mpimg.imread(BytesIO(r.content))
