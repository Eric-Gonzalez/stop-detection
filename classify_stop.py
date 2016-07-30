import glob
import os

import numpy as np
import scipy
import tflearn

from classifier.network import stop_dnn

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

# Convolutional Network
network = stop_dnn()

# Train using Classifier
# validation_set=(None, None),
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='classifier.tfl.ckpt')
model.fit(X_train, Y_train, n_epoch=50, shuffle=True,
          show_metric=True, batch_size=96, run_id='stop_cnn')
model.save('classifier.tfl')
