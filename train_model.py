
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

#Label va datani pustoy listga yuklaymiz.
data = []
labels = []

# Rasm boylab loop qilamiz
for imagePath in paths.list_images(args["dataset"]):

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = preprocess(image, 28, 28)
	image = img_to_array(image)
	data.append(image)

	#rasm pathdan labelini extract qilamiz.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Normalization qilamiz
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#Trainga 75 , testga 25 % ajratamiz
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

#Labellarni integerdan vektor ga convert qilamiz. .One-hot encoding
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# Modelni qurish
print("INFO compiling model...")

model = Sequential()
inputShape = (28, 28, 1)

# CONV => RELU => POOL layers
model.add(Conv2D(20, (5, 5), padding="same",
				 input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# CONV => RELU => POOL layers
model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(9))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer='SGD',
	metrics=["accuracy"])

# train the network
print("INFO training network...")
H = model.fit(trainX, trainY,  validation_data=(testX, testY),
	batch_size=32, epochs=15, verbose=1)

# evaluate the network
print(" INFO evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Modelni saqlash
model.save(args["model"])

# Train va Lossni plot qilish
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()