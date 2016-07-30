# Load Model
import numpy as np
import scipy
import tflearn
from classifier.network import stop_dnn

img = scipy.ndimage.imread('stop.png', mode="RGB")
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
data = np.asarray(img, dtype="int32")

network = stop_dnn()
model = tflearn.DNN(network)
model.load('classifier.tfl')

# Run Prediction
prediction = model.predict([data])
print prediction
