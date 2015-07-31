# Currently for testing

import dataset_manager
from captcha_recognizer import CaptchaRecognizer
import captcha_source


def main():
    # test_fetch_training_set()
    test_captcha_recognizing()


def test_fetch_training_set():
    dataset_manager.clear_training_set()
    dataset_manager.fetch_training_set(5)


def test_captcha_recognizing():
    image = captcha_source.fetch_image()
    CaptchaRecognizer().recognize(image)


if __name__ == '__main__':
    main()
