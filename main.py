import config as c
import dataset_manager
from captcha_recognizer import CaptchaRecognizer
from captcha_provider import BilibiliCaptchaProvider

# Below three imports is necessary for loading the pickle file, DO NOT DELETE
from captcha_learn import MLP, LogisticRegression, HiddenLayer


def main():
    dataset_manager.fetch_training_set(20)
    # test_captcha_recognition()
    dataset_manager.partition_training_images_to_chars()
    # dataset_manager.partition_training_images_to_chars(force_update=True,
    # save=True)
    # dataset_manager.tune_partition_parameter()
    # print(os.getcwd())
    # open(os.path.join(c.get('dataset'), 'best_model.pkl'), 'rb')


def test_captcha_recognition():
    c.clear_temp()
    seq = None

    # Below are all the training images that are partitioned falsely
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

    # image = dataset_manager.get_training_image(seq)
    # image = dataset_manager.get_test_image(seq)
    a = BilibiliCaptchaProvider()
    image = a.fetch()

    success, captcha = CaptchaRecognizer().recognize(image,
                                                     save_intermediate=True,
                                                     verbose=True,
                                                     reoptimize=False)
    if success:
        print(captcha)
        print('Recognized captcha is ', BilibiliCaptchaProvider().verify(captcha))



if __name__ == '__main__':
    main()
