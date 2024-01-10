from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import keras.backend as K
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
import math
import augmentation as au


def weighted_categorical_crossentropy(weights):
    weights = K.constant(weights, dtype='float32')
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        return -K.sum(loss, -1)
    return loss

def predict_with_single_input(model, image):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    dummy_roi = np.zeros((1, 224, 224, 1)) 
    circlePreds, labelPreds = model.predict([image, dummy_roi])
    return circlePreds, labelPreds



path = "Data/data.csv"
df = au.read_data(path)

#df_mirrored = au.blacked_background(df)

imagePaths = []

for index, row in df.iterrows():
	imagePath = row['path']
	image = cv2.imread(imagePath)
	#print(image.shape)
	imagePaths.append(imagePath)

weights_array_isAbnormal = np.load('Data/weights_array_isAbnormal.npy')

#print(weights_array_isAbnormal[1])
weights_array = np.load('Data/weights_array.npy')
#print(weights_array)

# model = load_model("Data/model/detectorVgg16.h5", custom_objects={'loss': weighted_categorical_crossentropy(weights_array)})
model = load_model("Data/model/detectorResNet50.h5", custom_objects={'loss': weighted_categorical_crossentropy(weights_array)})
lb = pickle.loads(open('Data/labels/lb.pickle', "rb").read())

model_is_abnormal = load_model("Data/model/detector_is_abnormal.h5", custom_objects={'loss': weighted_categorical_crossentropy(weights_array_isAbnormal)})
lb_is_abnormal = pickle.loads(open('Data/labels/lb_is_abnormal.pickle', "rb").read())


for imagePath in imagePaths:
	print(imagePath)

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	labelPred = model_is_abnormal.predict(image)
	i = np.argmax(labelPred, axis=1)
	label = lb_is_abnormal.classes_[i][0]
	label=1
	print(label)

	if label == 0:
		image = cv2.imread(imagePath)
		(h, w) = image.shape[:2]
		print(h, w, label)
		cv2.putText(image, "NORM", (int(h*0.75), int(w*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

	else:
		#circlePreds, labelPreds = predict_with_single_input(model, image)
		(circlePreds, labelPreds) = model.predict(image)
		print(circlePreds[0])
		# print("Wymiar", circlePreds.shape[1])
		(startX, startY, radius) = circlePreds[0][:3]
		i = np.argmax(labelPreds, axis=1)
		label = lb.classes_[i][0]

		image = cv2.imread(imagePath)
		(h, w) = image.shape[:2]
		print(h, w)
		print(startX, startY, radius)
		print(label)

		cv2.putText(image, label, (int(h * 0.1), int(w * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
		x = int(startX * w)
		startY = int(startY * h)
		radius = int(radius * w/5)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.circle(image, (x, y), radius, (0, 0, 255), 3)
		print(x, y, radius)


	cv2.imshow(imagePath, image)
	cv2.waitKey(0)
