import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import augmentation as au
#from augmentation import read_data, abnormality_masking
import pickle
import tensorflow
from keras.utils import to_categorical
from keras.applications import VGG16, ResNet50
from keras.layers import Flatten, Dropout, Dense, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imutils import paths
import keras.backend as K

def weighted_categorical_crossentropy(weights):
    # Convert weights to a Keras/TensorFlow constant of type float32
    weights = K.constant(weights, dtype='float32')
    def loss(y_true, y_pred):
        # Convert y_true to the same type as weights
        y_true = K.cast(y_true, 'float32')

        # scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculate loss and weight it
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss



path = "Data/data.csv"
df = au.read_data(path)
df = au.get_all_augmentation(df)
#df = au.blacked_background(df)
#df = au.abnormality_masking(df)
df = df.dropna(subset=['x'])
df.to_csv("dataframe_vgg16.csv")

# path = "dataframe_vgg16.csv.csv"
# df = pd.read_csv(path, sep=",")

#df = au.abnormality_masking(df)
#print(df.head(10))

# majority_class = 'NORM'
# majority_df = df[df['abnormality'] == majority_class]
# minority_df = df[df['abnormality'] != majority_class]
# # = df[df['abnormality'] != majority_class]
# #df = df.dropna(subset=['x'])
# #print(minority_df)
# #df.to_csv("Data/abnormalities.csv")
# desired_ratio = 0.5
# num_majority_instances = int(len(minority_df) * desired_ratio)
# undersampled_majority_df = majority_df.sample(n=num_majority_instances, random_state=42)

# undersampled_df = pd.concat([undersampled_majority_df, minority_df])
# undersampled_df = undersampled_df.reset_index()
# undersampled_df = undersampled_df.sort_values('index')
# undersampled_df.drop(['index'], axis='columns', inplace=True)
# undersampled_df = undersampled_df.reset_index()
# undersampled_df.drop(['index'], axis='columns', inplace=True)
# undersampled_df.to_csv("Data/balanced.csv")
# print(undersampled_df.head(10))
#
# df_balanced = undersampled_df
# df_balanced = df_balanced.dropna(subset=['x'])
#df_balanced.fillna(0, inplace=True)

print(df.head(10))
print(df['abnormality'].value_counts())

masks = []
imagePaths = []
data = []
labels = []
circles = []

for index, row in df.iterrows():
    label = row['abnormality']
    x = row['x']
    y = row['y']
    radius = row['radius']
    imagePath = row['path']
    #maskPath = row['mask_path']

    #print(x, y , radius)

    # mask = load_img(maskPath, target_size=(224, 224))
    # mask = img_to_array(mask) / 255.0
    image = cv2.imread(imagePath)
    #print(image.shape)
    (h, w) = image.shape[:2]
    x = float(x) / w
    y = float(y) / h
    radius = float(radius) / w

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    data.append(image)
    #masks.append(mask)
    labels.append(label)
    #print(x, y, radius)
    circles.append((x, y, radius))
    imagePaths.append(imagePath)

data = np.array(data, dtype="float32") / 255.0
masks = np.array(masks, dtype="float32")
labels = np.array(labels)
circles = np.array(circles, dtype="float32")
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)

split = train_test_split(data, labels, circles, imagePaths, test_size=0.2, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainCircles, testCircles) = split[4:6]
(trainPaths, testPaths) = split[6:]

f = open("Data/test_images_vgg16.txt", "w")
f.write("\n".join(testPaths))
f.close()

total_samples = trainLabels.shape[0]
class_totals = trainLabels.sum(axis=0)
class_weights = {i: total_samples / class_totals[i] for i in range(len(class_totals))}



# Build the model
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(256, activation="relu")(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

concat = Concatenate()([flatten, softmaxHead])

circleHead = Dense(128, activation="relu")(concat)
circleHead = Dense(64, activation="relu")(circleHead)
circleHead = Dense(32, activation="relu")(circleHead)
circleHead = Dense(3, activation="sigmoid", name="bounding_circle")(circleHead)

# Circle head
# circleHead = Dense(128, activation="relu")(flatten)
# circleHead = Dense(64, activation="relu")(circleHead)
# circleHead = Dense(32, activation="relu")(circleHead)
# circleHead = Dense(3, activation="sigmoid",
#                    name="bounding_circle")(circleHead)
# fully-connected layer head for class
# softmaxHead = Dense(512, activation="relu")(flatten)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(512, activation="relu")(softmaxHead)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(len(lb.classes_), activation="softmax",
# 	name="class_label")(softmaxHead)

model = Model(
	inputs=vgg.input,
	outputs=(circleHead, softmaxHead))

weights_array = np.array([class_weights[i] for i in range(len(class_weights))], dtype='float32')
np.save('Data/weights_array.npy', weights_array)

losses = {
    "class_label": weighted_categorical_crossentropy(weights_array),
    "bounding_circle": "mean_squared_error",
}

lossWeights = {
	"class_label": 1.0,
	"bounding_circle": 1.0
}


opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())


trainTargets = {
	"class_label": trainLabels,
	"bounding_circle": trainCircles
}

testTargets = {
	"class_label": testLabels,
	"bounding_circle": testCircles
}

# Train the model
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
    batch_size=32,
    epochs=30,
    verbose=1)


model.save("Data/model/detectorVgg16.h5", save_format="h5")

f = open('Data/labels/lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


lossNames = ["loss", "class_label_loss", "bounding_circle_loss"]
N = np.arange(0, 30)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

plt.tight_layout()
plotPath = os.path.sep.join(['Data/plots', "losses.png"])
plt.savefig(plotPath)
plt.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
    label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
    label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plotPath = os.path.sep.join(['Data/plots', "accs.png"])
plt.savefig(plotPath)




