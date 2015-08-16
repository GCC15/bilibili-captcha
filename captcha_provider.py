# Handle captcha fetching and providing

from abc import ABCMeta, abstractmethod
from io import BytesIO
import random

import requests
import matplotlib.image as mpimg

from helper import show_image


class HttpCaptchaProvider:
    """
    A HttpCaptchaProvider provides a state machine interface.
    It is assumed that "fetch" and "verify" can be done through HTTP/HTTPS urls.
    You can
    1. Fetch a CAPTCHA image;
    2. Verify if an answer to the last fetched CAPTCHA is correct.
    """
    __metaclass__ = ABCMeta

    def __init__(self, fetch_method, fetch_url, fetch_headers,
                 verify_method, verify_url, verify_headers):
        self.__fetch_method = fetch_method
        self.__fetch_url = fetch_url
        self.__fetch_headers = fetch_headers
        self.__verify_method = verify_method
        self.__verify_url = verify_url
        self.__verify_headers = verify_headers
        self.__session = requests.session()
        self.__is_virgin = True

    def fetch(self, retry_limit=3):
        self.__is_virgin = False
        print('Fetching CAPTCHA from {}'.format(self.__fetch_url))
        r = None
        for num_retries in range(retry_limit):
            if num_retries > 0:
                print('Retry: {}'.format(num_retries))
            try:
                r = self.__session.request(
                    self.__fetch_method,
                    self.__fetch_url,
                    headers=self.__fetch_headers
                )
                break
            except Exception as e:
                print('An exception occurs when fetching captcha', e)
        return None if r is None else mpimg.imread(BytesIO(r.content))

    def verify(self, seq):
        if self.__is_virgin:
            raise ValueError('Must fetch a CAPTCHA first!')
        r = self.__session.request(
            self.__verify_method,
            self.__verify_url,
            headers=self.__verify_headers,
            data=self._get_data_from_seq(seq)
        )
        return self._is_correct_response(r)

    @abstractmethod
    def _get_data_from_seq(self, seq):
        raise NotImplementedError()

    @abstractmethod
    def _is_correct_response(self, r):
        raise NotImplementedError()


# A NormalSeqSet represents the set of all sequences with a fixed length, and
# with the chars chosen from a set.
class NormalSeqSet:
    def __init__(self, chars, seq_length):
        self.chars = chars
        self.charset = set(chars)
        self.seq_length = seq_length

    def canonicalize_seq(self, seq):
        return seq

    def is_valid_seq(self, seq):
        return (len(seq) == self.seq_length and
                all(char in self.charset for char in seq))


# BilibiliCaptchaProvider provides captcha from www.bilibili.com
class BilibiliCaptchaProvider(HttpCaptchaProvider, NormalSeqSet):
    __GET = 'GET'
    __POST = 'POST'
    __USER_AGENT = 'User-Agent'
    __HOST = 'Host'
    __REFERER = 'Referer'

    __user_agent = ('Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) '
                    'Gecko/20100101 Firefox/34.0')
    __host = 'account.bilibili.com'

    __verify_method = __POST
    __verify_url = 'https://account.bilibili.com/register/mail'
    __verify_headers = {
        __USER_AGENT: __user_agent,
        __HOST: __host,
        __REFERER: __verify_url
    }

    __fetch_method = __GET
    __fetch_url = 'https://www.bilibili.com/captcha'
    __fetch_headers = {
        __USER_AGENT: __user_agent,
        __HOST: __host,
        __REFERER: __verify_url
    }

    def __init__(self):
        #       'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        chars = '    EFGH JKLMN PQR TUVWXY  123456 89'.replace(' ', '')
        NormalSeqSet.__init__(self, chars, 5)
        HttpCaptchaProvider.__init__(
            self,
            self.__fetch_method, self.__fetch_url, self.__fetch_headers,
            self.__verify_method, self.__verify_url, self.__verify_headers,
        )

    def verify(self, seq):
        return self.is_valid_seq(seq) and HttpCaptchaProvider.verify(self, seq)

    def _get_data_from_seq(self, seq):
        return {'vd': seq, 'action': "checkVd"}

    def _is_correct_response(self, r):
        r_json = r.json()
        if r_json['status']:
            return True
        else:
            print('The response of verifying captcha is', r_json['message'])
            return False

    def canonicalize_seq(self, seq):
        return seq.upper()


def _test_bilibili():
    captcha_provider = BilibiliCaptchaProvider()
    print(captcha_provider.is_valid_seq(
        ''.join(random.sample(captcha_provider.chars, 5))
    ))
    print(captcha_provider.is_valid_seq(
        ''.join(random.sample(captcha_provider.chars, 4)) + ' '
    ))
    show_image(captcha_provider.fetch())
    seq = captcha_provider.canonicalize_seq(
        input('Input the answer sequence to verify: ')
    )
    print(captcha_provider.verify(seq))


def main():
    _test_bilibili()


if __name__ == '__main__':
    main()
