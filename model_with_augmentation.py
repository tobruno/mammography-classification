import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import augmentation as au
import pickle
import tensorflow
from keras.utils import to_categorical
from keras.applications import ResNet50, VGG16
from keras.layers import Flatten, Dropout, Dense, Input, Reshape, Resizing, Concatenate
from keras.layers import Conv2D, Add, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Cropping2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.initializers import Zeros
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
# # df = au.get_all_augmentation(df)
# df = au.abnormality_masking(df)
# df = df.dropna(subset=['x'])
# # df.to_csv("dataframe.csv")

path = "dataframe.csv"
df = pd.read_csv(path, sep=",")
print(df.head(10))




print("[INFO] loading dataset...")
masks = []
imagePaths = []
maskPaths = []
data = []
labels = []
circles = []

for index, row in df.iterrows():
    label = row['abnormality']
    x = row['x']
    y = row['y']
    radius = row['radius']
    imagePath = row['path']
    maskPath = row['mask_path']

    mask = load_img(maskPath, color_mode='grayscale', target_size=(224, 224))
    mask = img_to_array(mask)

    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # print(x, y, radius)
    # print(image.shape)
    x = float(x) / w
    y = float(y) / h
    radius = (float(radius)*5) / w

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    data.append(image)
    masks.append(mask)
    labels.append(label)
    #print(x, y, radius)
    circles.append((x, y, radius))
    imagePaths.append(imagePath)
    maskPaths.append(maskPath)

data = np.array(data, dtype="float32") / 255.0
masks = np.array(masks, dtype="float32") / 255.0
labels = np.array(labels)
circles = np.array(circles, dtype="float32")
imagePaths = np.array(imagePaths)
maskPaths = np.array(maskPaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(len(lb.classes_))
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

# Split the data
split = train_test_split(data, labels, circles, imagePaths, masks, maskPaths, test_size=0.2, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainCircles, testCircles) = split[4:6]
(trainPaths, testPaths) = split[6:8]
(trainMasks, testMasks) = split[8:10]
(trainMaskPaths, testMaskPaths) = split[10:]

total_samples = trainLabels.shape[0]
class_totals = trainLabels.sum(axis=0)
class_weights = {i: total_samples / class_totals[i] for i in range(len(class_totals))}

# # ResNet architecture
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# base_model.trainable = False
#
# roi_input = Input(shape=(224, 224, 1))
# attention_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(roi_input)
# attention_branch = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(attention_branch)

# attention_branch = Conv2DTranspose(filters=2048, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(attention_branch)
# attention_branch = Cropping2D(cropping=((0, 1), (0, 1)))(attention_branch)
# # Resize to the exact dimensions of the base_model output
# attention_branch = Resizing(height=7, width=7)(attention_branch)
#
# combined_layer = Add()([base_model.output, attention_branch])
# combined_output = GlobalAveragePooling2D()(combined_layer)
#
# flatten = Flatten()(combined_output)
# classHead = Dense(512, activation="relu")(flatten)
# classHead = Dropout(0.5)(classHead)
# classHead = Dense(512, activation="relu")(classHead)
# classHead = Dropout(0.5)(classHead)
# class_output = Dense(len(lb.classes_), activation="softmax", name="class_label")(classHead)
#
# concat = Concatenate()([flatten, class_output])
#
# circleHead = Dense(128, activation="relu")(concat)
# circleHead = Dense(64, activation="relu")(circleHead)
# circleHead = Dense(32, activation="relu")(circleHead)
# circle_output = Dense(3, activation="sigmoid", name="bounding_circle")(circleHead)

# ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# roi_input = Input(shape=(224, 224, 1))
# # Single Convolution Layer with Zeros Initialization
# attention_branch = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), activation='relu',
#                           kernel_initializer=Zeros(), bias_initializer=Zeros())(roi_input)
# attention_branch = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(attention_branch)
# attention_branch = Resizing(height=28, width=28)(attention_branch)
#
# resnet_layer_to_combine = base_model.get_layer('conv3_block4_out').output # Example layer index
#
# combined_layer = Add()([resnet_layer_to_combine, attention_branch])
#
# combined_output = GlobalAveragePooling2D()(combined_layer)

flatten = base_model.output
flatten = Flatten()(flatten)

classHead = Dense(1024, activation="relu")(flatten)
classHead = Dropout(0.5)(classHead)
classHead = Dense(512, activation="relu")(classHead)
classHead = Dropout(0.5)(classHead)
classHead = Dense(256, activation="relu")(classHead)
class_output = Dense(len(lb.classes_), activation="softmax", name="class_label")(classHead)

#concat = Concatenate()([flatten, class_output])

circleHead = Dense(256, activation="relu")(flatten)
circleHead = Dropout(0.5)(circleHead)
circleHead = Dense(128, activation="relu")(circleHead)
circleHead = Dropout(0.5)(circleHead)
circleHead = Dense(64, activation="relu")(circleHead)
circleHead = Dense(32, activation="relu")(circleHead)
circle_output = Dense(3, activation="sigmoid", name="bounding_circle")(circleHead)

model = Model(inputs=base_model.input, outputs=(circle_output, class_output))

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

# Compile the model
model.compile(optimizer=Adam(lr=1e-4),
              loss={'class_label': 'categorical_crossentropy', 'bounding_circle': 'mean_squared_error'},
              metrics=["accuracy"],
              loss_weights=lossWeights)
print(model.summary())

trainTargets = {
	"class_label": trainLabels,
	"bounding_circle": trainCircles
}

testTargets = {
	"class_label": testLabels,
	"bounding_circle": testCircles
}

# checkpointPath = 'Data/model/model_{epoch:02d}-{val_loss:.2f}.h5'
# checkpoint = ModelCheckpoint(checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)


# Train the model
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
    batch_size=16,
    epochs=30,
    verbose=1,
    #callbacks=[checkpoint]
)


model.save("Data/model/detectorResNet50.h5", save_format="h5")

f = open('Data/labels/lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


lossNames = ["loss", "class_label_loss", "bounding_circle_loss"]
N = np.arange(0, 30)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

plt.tight_layout()
plotPath = os.path.sep.join(['Data/plots', "lossesResNet.png"])
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
plotPath = os.path.sep.join(['Data/plots', "accsResNet.png"])
plt.savefig(plotPath)
