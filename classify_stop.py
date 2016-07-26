import glob
import os
import scipy

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from PIL import Image, ImageDraw, ImageFont, ImageOps

# TODO Resize every image to 32x32

# X is the Training Data
# Y is the Labels
X_train = []
Y_train = []


def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def resize_image(filename):
    img = scipy.ndimage.imread(filename, mode="RGB")
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
    return np.asarray(img, dtype="int32")


# Load Stop Sign Images
for index, filename in enumerate(glob.glob(os.path.join('stop-annotations', '*.png'))):
    X_train.append(resize_image(filename))
    Y_train.append([1, 0])

for index, filename in enumerate(glob.glob(os.path.join('not-stop-annotations', '*.png'))):
    X_train.append(resize_image(filename))
    Y_train.append([0, 1])


X_train, Y_train = shuffle(X_train, Y_train)

# X Test is Test Training Data
# Y Test is Test Training Labels

# Data loading and pre-processing
img_pre_processing = ImagePreprocessing()
# img_pre_processing.add_featurewise_stdnorm()
# img_pre_processing.add_featurewise_zero_center()


# Data Augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=10.)

# Convolutional Network
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

# Train using Classifier
# validation_set=(None, None),
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='classifier.tfl.ckpt')
model.fit(X_train, Y_train, n_epoch=50, shuffle=True,
          show_metric=True, batch_size=96, run_id='stop_cnn')
model.save('classifier.tfl')
