from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

imagePaths = sorted(list(paths.list_images("training_data")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image
    image = cv2.imread(imagePath)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("model.h5")

    # classify the input image
    (negative, positive) = model.predict(image)[0]

    # build the label
    label = "Positive" if positive > negative else "Negative"
    proba = positive if positive > negative else negative
    label = "{}: {:.2f}%".format(label, proba * 100)

    trueLabel = imagePath.split(os.path.sep)[-2]
    trueLabel = "Actually Positive" if trueLabel == "positive" else "Actually Negative"

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    	0.7, (0,255,0), 2)
    cv2.putText(output, trueLabel, (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0,255,0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
