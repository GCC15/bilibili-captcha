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
    # test_recognize_training()
    # dataset_manager.fetch_training_set(50)
    # test_recognize_training()
    # captcha_learn.reconstruct_model()
    # test_recognize_http(num=30)
    # dataset_manager.get_training_images(1)
    # dataset_manager.partition_training_images_to_chars()
    # dataset_manager.partition_training_images_to_chars(force_update=True,
    # save=True)
    # dataset_manager.tune_partition_parameter()


def test_recognize_training():
    c.clear_temp()
    seq = 'JWP26'
    # seq = 'K464J'

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
    success, seq_r, weak_confidence = CaptchaRecognizer().recognize(image,
                                                                    verbose=True,
                                                                    save_intermediate=True,
                                                                    force_partition=True)
    if success:
        if weak_confidence:
            print('Weak confidence')
        print('Recognized is', seq_r)
        print('Actual is', seq)
        print('Result: {}'.format(seq == seq_r))


def test_recognize_http(show_img=False, num=1, reconstruct=False,
                        force_partition=True):
    time_start = time.time()
    provider = BilibiliCaptchaProvider()
    recognizer = CaptchaRecognizer()
    fail = 0
    right_strong = 0
    right_weak = 0
    wrong_strong = 0
    wrong_weak = 0
    for i in range(num):
        image = time_func(
            'fetch' if num == 1 else None,
            lambda: provider.fetch()
        )
        if show_img and num == 1:
            show_image(image)
        if num == 1:
            success, seq, weak_confidence = recognizer.recognize(image,
                                                                 save_intermediate=True,
                                                                 verbose=True,
                                                                 reconstruct=reconstruct,
                                                                 force_partition=force_partition)
        else:
            if i == 0:
                success, seq, weak_confidence = recognizer.recognize(image,
                                                                     save_intermediate=False,
                                                                     verbose=False,
                                                                     reconstruct=reconstruct,
                                                                     force_partition=force_partition)
            else:
                success, seq, weak_confidence = recognizer.recognize(image,
                                                                     save_intermediate=False,
                                                                     verbose=False,
                                                                     reconstruct=False,
                                                                     force_partition=force_partition)
        if success:
            print(seq)
            result = time_func(
                'verify' if num == 1 else None,
                lambda: provider.verify(seq)
            )
            if num == 1:
                print('Recognized seq is {}'.format(result))
            if result:
                if weak_confidence:
                    right_weak += 1
                else:
                    right_strong += 1
            else:
                if weak_confidence:
                    wrong_weak += 1
                else:
                    wrong_strong += 1
        else:
            fail += 1
    right_total = right_strong + right_weak
    wrong_total = wrong_strong + wrong_weak
    print('Fail: ', fail)
    print('Right weak: ', right_weak)
    print('Right strong: ', right_strong)
    print('Right total: ', right_total)
    print('Wrong weak: ', wrong_weak)
    print('Wrong strong: ', wrong_strong)
    print('Wrong total: ', wrong_total)
    print('Total success rate: ', (right_weak + right_strong) / num)
    print('Success rate when confident: ',
          (right_strong + right_weak) / (num - fail) if num - fail > 0 else 0)
    print('Success rate when strongly confident: ',
          right_strong / (
              right_strong + wrong_strong) if right_strong + wrong_strong > 0
          else 0)
    print('Success rate when weakly confident: ',
          right_weak / (
              right_weak + wrong_weak) if right_weak + wrong_weak > 0
          else 0)
    time_end = time.time()
    print('Time used to test recognize http is: ', time_end - time_start)


if __name__ == '__main__':
    main()
