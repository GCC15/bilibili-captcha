# Currently for testing

import config as c
import dataset_manager
from captcha_recognizer import CaptchaRecognizer
import captcha_source


def main():
    # test_fetch_training_set()
    test_captcha_recognizing()


def test_fetch_training_set():
    # dataset_manager.clear_training_set()
    dataset_manager.fetch_training_set(50)


def test_captcha_recognizing():
    c.clear_temp()
    # image = captcha_source.fetch_image()
    image = dataset_manager.get_training_images(1)[0]
    # image = dataset_manager.get_training_image('J11L2')
    # image = dataset_manager.get_training_image('EQEJU')
    # image = dataset_manager.get_training_image('QN4EL')
    CaptchaRecognizer().recognize(image)


if __name__ == '__main__':
    main()
