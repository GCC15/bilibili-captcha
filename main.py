# Currently for testing

import dataset_manager


def main():
    test_fetch_training_set()


def test_fetch_training_set():
    dataset_manager.clear_training_set()
    dataset_manager.fetch_training_set(5)


if __name__ == '__main__':
    main()
