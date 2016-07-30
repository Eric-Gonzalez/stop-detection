from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def stop_dnn():
    img_pre_processing = ImagePreprocessing()

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=10.)

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_pre_processing,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network
