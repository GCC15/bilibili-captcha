# The place to run the program and main place for testing

import time

import config as c
import dataset_manager
from captcha_recognizer import CaptchaRecognizer
from captcha_provider import BilibiliCaptchaProvider
from helper import show_image, time_func
# noinspection PyUnresolvedReferences
import captcha_learn


def main():
    pass
    # dataset_manager.fetch_training_set(50)
    # test_recognize_training()
    # captcha_learn.reconstruct_model()
    # test_recognize_http()
    test_recognize_http(num=100)
    # dataset_manager.get_training_images(1)
    # dataset_manager.partition_training_images_to_chars()
    # dataset_manager.partition_training_images_to_chars(force_update=True,
    # save=True)
    # dataset_manager.tune_partition_parameter()


def test_recognize_training():
    c.clear_temp()
    seq = None

    # Below are sample training images that are partitioned falsely
    # seq = 'YFF5M'
    # seq = 'W1PM4'
    # seq = 'W1R4R'
    # seq = 'YTM6X'
    # seq = 'W9WU4'
    # seq = 'EFTWY'
    # seq = '5WTGP'
    # seq = '113W2'
    # seq = 'UWFG1'

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

    if seq:
        image = dataset_manager.get_training_image(seq)
    else:
        seq, image = dataset_manager.get_training_image()
    success, seq_r = CaptchaRecognizer().recognize(image, verbose=True,
                                                   save_intermediate=True)
    if success:
        print('Result: {}'.format(seq == seq_r))
        # image = dataset_manager.get_test_image(seq)


def test_recognize_http(show_img=False, num=1, reconstruct=False):
    time_start = time.time()
    provider = BilibiliCaptchaProvider()
    recognizer = CaptchaRecognizer()
    fail = 0
    right = 0
    wrong = 0
    for i in range(num):
        image = time_func(
            'fetch' if num == 1 else None,
            lambda: provider.fetch()
        )
        if show_img and num == 1:
            show_image(image)
        if num == 1:
            success, seq = recognizer.recognize(image,
                                                save_intermediate=True,
                                                verbose=True,
                                                reconstruct=reconstruct)
        else:
            if i == 0:
                success, seq = recognizer.recognize(image,
                                                    save_intermediate=False,
                                                    verbose=False,
                                                    reconstruct=reconstruct)
            else:
                success, seq = recognizer.recognize(image,
                                                    save_intermediate=False,
                                                    verbose=False,
                                                    reconstruct=False)
        if success:
            print(seq)
            result = time_func(
                'verify' if num == 1 else None,
                lambda: provider.verify(seq)
            )
            if num == 1:
                print('Recognized seq is {}'.format(result))
            if result:
                right += 1
            else:
                wrong += 1
        else:
            fail += 1
    print('Fail: ', fail)
    print('Right: ', right)
    print('Wrong: ', wrong)
    print('Total success rate: ', right / num)
    print('Success rate when confident: ',
          right / (right + wrong) if right + wrong > 0 else 0)
    time_end = time.time()
    print('Time used to test recognize http is: ', time_end - time_start)


if __name__ == '__main__':
    main()
