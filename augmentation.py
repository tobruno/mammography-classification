import pandas as pd
import numpy as np
import glob
import os
import cv2
import imutils
import random

import scipy
#import seaborn as sns
import math


#path = "Data/data.csv"

def read_data(path):
    df = pd.read_csv(path, sep=" ")
    df['path'] = df['name'].apply(lambda x: f"Data/images/{x}.pgm")
    df = df[df.abnormality != 'ASYM']
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
    return df

#df = read_data(path)
#print(df.head(10))

# df1=df
# df1.tissue = pd.factorize(df1.tissue)[0]
# df1.abnormality = pd.factorize(df1.abnormality)[0]
# df1.severity = pd.factorize(df1.severity)[0]
# df1 = df1.drop(['name','x', 'y'], axis=1)
# ax = sns.heatmap(df1.corr(), annot=True)
# plt.show()

def rotate_coordinates(x, y, center_x, center_y, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    x_rotated = (x - center_x) * np.cos(angle_radians) - (y - center_y) * np.sin(angle_radians) + center_x
    y_rotated = (x - center_x) * np.sin(angle_radians) + (y - center_y) * np.cos(angle_radians) + center_y
    return x_rotated, y_rotated
def rotation(image, angle):
    center = tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

def rotate_image(df1, path, case):
# Open the image
    df = df1.copy()
    rotated_path_list = []
    rotated_name_list = []
    rotated_x_list = []
    rotated_y_list = []

    image_paths = df['path'].tolist()
    image_names = df['name'].tolist()
    image_x, image_y = df['x'].tolist(), df['y'].tolist()
    duplicates = ""
    for i in range(len(image_paths)):
        #image = Image.open(image_paths[i])
        base_image_path = image_paths[i].rsplit('.', 1)[0]
        image = cv2.imread(image_paths[i])
        if len(image.shape) == 2:
            print("Image is already in grayscale format.")
        else:
            #print("Image is not in grayscale format.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Original Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]

# Rotation parameters
        center_x, center_y = width / 2, height / 2
        rotation_angle = random.randint(-20, 20)  # Replace with your desired rotation angle
# Rotate the image
        #rotated_image = image.rotate(rotation_angle, center=(center_x, center_y))
        rotated_image = rotation(image, rotation_angle)
        if case == 1:
            rotated_image = blur_image(rotated_image)
        if case == 2:
            rotated_image = sharpen_image(rotated_image)
        if case == 3:
            rotated_image = enhance_image(rotated_image)
        if case == 4:
            rotated_image = equalize_image(rotated_image)

        if image_names[i-1] == image_names[i]:
            rotated_name = image_names[i] + "_rotated1"
        if image_names[i - 2] == image_names[i]:
            rotated_name = image_names[i] + "_rotated2"
        else:
            rotated_name = image_names[i] + "_rotated"
        rotated_path = path.rsplit('.', 1)[0] + rotated_name  + ".pgm"

# Save or display the rotated image
        cv2.imwrite(rotated_path, rotated_image)
        # cv2.imshow('Original Image', rotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        original_x, original_y = image_x[i], image_y[i] # Replace with the initial coordinates of your pixel
        rotated_x, rotated_y = rotate_coordinates(original_x, original_y, center_x, center_y, rotation_angle)

        rotated_path_list.append(rotated_path)
        rotated_name_list.append(rotated_name)
        rotated_x_list.append(rotated_x)
        rotated_y_list.append(rotated_y)

        # print(f"Original Coordinates: ({original_x}, {original_y})")
        # print(f"Rotated Coordinates: ({rotated_x}, {rotated_y})")

    df['path'] = rotated_path_list
    df['x'] =  list(map(float, rotated_x_list))
    df['y'] =  list(map(float, rotated_y_list))

    return df


def mirror_image(df1, path, case):
    df = df1.copy()
    x = df['x'].tolist()
    image_paths = df['path'].tolist()
    image_names = df['name'].tolist()
    x_mirrored_list = []
    mirrored_name_list = []
    mirrored_path_list = []
    for i in range(len(image_paths)):

        image = cv2.imread(image_paths[i])
        base_image_path = image_paths[i].rsplit('.', 1)[0]
        if len(image.shape) == 2:
            print("Image is already in grayscale format.")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (height, width) = image.shape[:2]
        image = cv2.flip(image, 1)
        if case == 1:
            image = blur_image(image)
        if case == 2:
            image = sharpen_image(image)
        if case == 3:
                image = enhance_image(image)
        if case == 4:
            image = equalize_image(image)

        if x != None:
            x_mirrored = width - x[i]
        else:
            x_mirrored = x
        x_mirrored_list.append(x_mirrored)
        if image_names[i - 1] == image_names[i]:
            mirrored_name = image_names[i] + "_mirrored1"
        if image_names[i - 2] == image_names[i]:
            mirrored_name = image_names[i] + "_mirrored2"
        else:
            mirrored_name = image_names[i] + "_mirrored"
        mirrored_path = path.rsplit('.', 1)[0] + mirrored_name + ".pgm"
        mirrored_name_list.append(mirrored_name)
        mirrored_path_list.append(mirrored_path)
        cv2.imwrite(mirrored_path, image)
    df['path'] = mirrored_path_list

    df['x'] = x_mirrored_list
    return df



def blur_image(image):
    blur = cv2.medianBlur(image, 5)
    # cv2.imshow('blur', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blur


#img = cv2.imread("Data/images/mdb001.pgm", cv2.IMREAD_GRAYSCALE)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    # cv2.imshow('AV CV- Winter Wonder Sharpened', sharp_image)
    # cv2.imshow('Original', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return sharp_image


image = cv2.imread("Data/images/mdb002.pgm")
# img = sharpen_image(image)


def enhance_image(image):
    if len(image.shape) == 2:
        pass# print("Image is already in grayscale format.")
    else:
        # print("Image is not in grayscale format.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Create a CLAHE object (Clip Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    enhanced_image = cv2.threshold(enhanced_image, 13, 255, cv2.THRESH_TOZERO)[1]
    # Apply CLAHE
    # Display the original and equalized images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Enhanced Image', enhanced_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return enhanced_image

def equalize_image(image):
    if len(image.shape) == 2:
        pass# print("Image is already in grayscale format.")
    else:
        # print("Image is not in grayscale format.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    equalized_image = cv2.equalizeHist(image)
    # Display the original and equalized images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Equalized Image', equalized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return equalized_image


#rotate_image()

def abnormality_masking(df):
    x = df['x'].tolist()
    y = df['y'].tolist()
    radius = df['radius'].tolist()
    x = [int(value) if not math.isnan(value) else value for value in x]
    y = [int(value) if not math.isnan(value) else value for value in y]
    radius = [int(value) if not math.isnan(value) else value for value in radius]
    image_paths = df['path'].tolist()
    image_names = df['name'].tolist()
    mask_name_list = []
    mask_path_list = []
    name_counts = {}
    for i in range(len(image_paths)):
        base_image_path = image_paths[i].rsplit('.', 1)[0]
        name_counts[image_names[i]] = name_counts.get(image_names[i], 0) + 1
        image = cv2.imread(image_paths[i])
        if len(image.shape) == 2:
            pass#print("Image is already in grayscale format.")
        else:
            #print("Image is not in grayscale format.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if not math.isnan(x[i]) and not math.isnan(y[i]):
            mask = np.zeros(image.shape, dtype="uint8")
            cv2.circle(mask, (x[i], y[i]), radius[i], (255), -1)
            # Apply mask by setting the pixels inside the circle to white
            image[mask == 255] = 255
        else:
            mask = np.zeros(image.shape, dtype="uint8")

        # The variable `masked` should contain the image with the circle in white
        masked = cv2.bitwise_and(image, image, mask=mask)
        if name_counts[image_names[i]] == 1:
            mask_path = f"{base_image_path}_mask.pgm"
        else:
            mask_path = f"{base_image_path}_mask_{name_counts[image_names[i]]-1}.pgm"
        mask_path_list.append(mask_path)
        cv2.imwrite(mask_path, masked)
    df['mask_path'] = mask_path_list

    return df

#show_image(img)

#image = cv2.imread("Data/images/mdb002.pgm")
def masking(image):
    if len(image.shape) == 2:
        # print("Image is already in grayscale format.")
        gray = image
    else:
        # print("Image is not in grayscale format.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    epsilon = 0.0018 * cv2.arcLength(c, True)
    smoothed_contour = cv2.approxPolyDP(c, epsilon, True)
# Create a blank image
    mask = np.zeros_like(gray)
# Draw the contour on the mask
    cv2.drawContours(mask, [smoothed_contour], -1, 255, thickness=cv2.FILLED)
# Display the original image with the contour and the mask
    result = cv2.bitwise_and(image, image, mask=mask)
#     cv2.imshow("Original Image", image)
#     cv2.imshow("Image with Contour", result)
#     cv2.imshow("Mask", mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return result

def blacked_background(df):
    image_paths = df['path'].tolist()
    image_names = df['name'].tolist()
    mask_name_list = []
    mask_path_list = []

    for i in range(len(image_paths)):
        # print(x[i])
        image = cv2.imread(image_paths[i])
        if len(image.shape) == 2:
            pass  # print("Image is already in grayscale format.")
        else:
            # print("Image is not in grayscale format.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = masking(image)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if image_names[i - 1] == image_names[i]:
            mask_name = image_names[i] + "_blacked1"
        if image_names[i - 2] == image_names[i]:
            mask_name = image_names[i] + "_blacked2"
        else:
            mask_name = image_names[i] + "_blacked"
        mask_path = "Data/no_background/" + mask_name + ".pgm"
        mask_name_list.append(mask_name)
        mask_path_list.append(mask_path)
        cv2.imwrite(mask_path, mask)
    df['path'] = mask_path_list

    return df


def get_all_augmentation(df):
    df_mirrored = mirror_image(df, path="Data/mirrored/", case=0)
    print(df_mirrored.head(10))
    df_rotated = rotate_image(df, path="Data/rotated/", case=0)
    print(df_rotated.head(10))
    df_mirrored_equalized = mirror_image(df, path="Data/mirrored_equalized/", case=4)
    df_rotated_blurred= rotate_image(df, path="Data/rotated_blurred/", case=1)
    df_mirrored_rotated_sharpened = rotate_image(df_mirrored, path="Data/mirrored_rotated_sharpened/", case=2)

    df_rotated1 = df_rotated.copy()
    df_rotated1 = blacked_background(df_rotated1)

    df_mirrored1 = df_mirrored.copy()
    df_mirrored1 = blacked_background(df_mirrored1)
    #df_blurred_enhanced = mirror_image(mirror_image(df, path="Data/enhanced_blurred/", case=3), path="Data/enhanced_blurred/", case=1)
    df1 = pd.concat([df, df_mirrored, df_rotated, df_mirrored_equalized, df_rotated_blurred, df_mirrored_rotated_sharpened, df_mirrored1, df_rotated1])
    return df1















# df = read_data(path)
#masking(equalize_image(image))    Odpada
#equalize_image(enhance_image(image))  To można maskować

# mask = masking(equalize_image(enhance_image(image)))
# masked = cv2.bitwise_and(image,image,mask = mask)
# cv2.imshow("Mask", masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#masking(image)


# df1 = abnormality_masking(df)
# print(df1.head(10))
#rotate_image(df)
# mirror_image(df)

# image = cv2.imread("Data/mirrored/mdb002.pgm")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# mask = abnormality_masking(image)
#
# cv2.imshow("Original Image", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()