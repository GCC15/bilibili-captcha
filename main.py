# Currently for testing

import config as c
import dataset_manager
from captcha_recognizer import CaptchaRecognizer
import captcha_source


def main():
    # dataset_manager.fetch_training_set(100)
    # test_captcha_recognition()
    # dataset_manager.partition_training_images_to_chars()
    # dataset_manager.partition_training_images_to_chars(force_update=True)
    # dataset_manager.tune_partition_parameter()
    captcha_source.fill_captcha()


def test_captcha_recognition():
    c.clear_temp()
    seq = None
    # seq = 'QN4EL'
    # seq = 'YFF5M'

    # Sticking together
    # seq = 'WMQPQ'
    # seq = '14FWX'
    # seq = '4TJ3R'
    # seq = '5PW9Y'
    # seq = '6ML6X'
    # seq = '48HXH'
    # seq = 'Y581K'

    # Isolation
    # seq = 'QN4EL'

    # Complicated
    # seq = '2XML9'
    # seq = 'W9WU4'

    image = dataset_manager.get_training_image(seq)
    CaptchaRecognizer().recognize(image, save_intermediate=True, verbose=True)


if __name__ == '__main__':
    main()
