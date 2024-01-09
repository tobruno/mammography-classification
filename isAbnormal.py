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
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
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


# path = "Data/data.csv"
# df = au.read_data(path)
# df = au.blacked_background(df)
# df = au.get_all_augmentation(df)
# df = au.abnormality_masking(df)
# df.to_csv("dataframe1.csv")
path = "dataframe1.csv"
df1 = pd.read_csv(path, sep=",")
df2 = df1.copy()
df = au.blacked_background(df2)
#print(df.head(10))
#df = df.dropna(subset=['abnormality'])
majority_class = 'NORM'


df['abnormality'] = df['abnormality'].apply(lambda x: 0 if x == 'NORM' else 1)

# majority_df = df[df['abnormality'] == 0]
# minority_df = df[df['abnormality'] != 0]
#
# desired_ratio = 0.9
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
# df_balanced = undersampled_df



data = []
labels = []
imagePaths = []

print(df.head(20))

for index, row in df.iterrows():
    label = row['abnormality']
    imagePath = row['path']

    image = load_img(imagePath, target_size=(224, 224))
    #
    image = img_to_array(image)


    data.append(image)
    labels.append(label)
    imagePaths.append(imagePath)

print(labels)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
imagePaths = np.array(imagePaths)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(len(lb.classes_))
if len(lb.classes_) == 2:
    labels = to_categorical(labels)



split = train_test_split(data, labels, imagePaths, test_size=0.2, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainPaths, testPaths) = split[4:6]


f = open("Data/test_images_is_abnormal.txt", "w")
f.write("\n".join(testPaths))
f.close()

total_samples = trainLabels.shape[0]
class_totals = trainLabels.sum(axis=0)
class_weights = {i: total_samples / class_totals[i] for i in range(len(class_totals))}
print(class_weights)

resnet = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
resnet.trainable = False
flatten = resnet.output
flatten = Flatten()(flatten)

flatten = Flatten()(flatten)
softmaxHead = Dense(1024, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(256, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(inputs=resnet.input, outputs=softmaxHead)

weights_array = np.array([class_weights[i] for i in range(len(class_weights))], dtype='float32')
np.save('Data/weights_array_isAbnormal.npy', weights_array)

# Define your model's loss functions
losses = {
    "class_label": weighted_categorical_crossentropy(weights_array),
}


opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"])
#print(model.summary())



trainTargets = {
    "class_label": trainLabels
}
testTargets = {
    "class_label": testLabels
}


print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=16,
    epochs=10,
    verbose=1)

model.save("Data/model/detector_is_abnormal.h5", save_format="h5")

f = open('Data/labels/lb_is_abnormal.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


lossName = "loss"
N = np.arange(0, 10)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(1, 1, figsize=(13, 13))

title = "Loss for presence of abnormality"
ax.set_title(title)
ax.set_xlabel("Epoch #")
ax.set_ylabel("Loss")
ax.plot(N, H.history["loss"], label="loss")
ax.plot(N, H.history["val_loss"], label="val_loss")
ax.legend()

plt.tight_layout()
plotPath = os.path.sep.join(['Data/plots', "losses_is_abnormal.png"])
plt.savefig(plotPath)
plt.close()



plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["accuracy"],
    label="accuracy")
plt.plot(N, H.history["val_accuracy"],
    label="val_accuracy")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join(['Data/plots', "accs_is_abnormal.png"])
plt.savefig(plotPath)