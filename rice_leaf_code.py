# -*- coding: utf-8 -*-
"""
Created on Fri May 12 01:44:01 2023

@author: Sayantan Chakraborty

"""

import pickle
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel, laplace, gaussian, meijering
from sklearn.model_selection import train_test_split
from rembg import remove
from datetime import datetime
from time import time
print(os.listdir("Diseases/"))
print(os.listdir("Nutrition_Defeciency/"))
SIZE = 256


# Data Variables
images = []
labels = []
train_images = []
train_labels = []
test_images = []
test_labels = []


# Categorical Prediction and Actual Labels
image_names = []
image_classes = []


# Global Variables for storing data in excel
n_est = 150
accuracy = 0
models = []
precision_score = 0
recall_score = 0
f1_score = 0
random_state = 50
class_report = [[]]
class_data = pd.DataFrame()
# -------------------------------------------

# -----------------------------------------Images to List-----------------------------------------
for path in glob.glob("Nutrition_Defeciency/*"):
    label = path.split("\\")[-1]
    print(label)
    for i in range(0, 50):
        # for img_path in glob.glob(os.path.join(path, "*.jpg")):
        img_path = glob.glob(os.path.join(path, "*.JPG"))[i]
        filename = os.path.basename(img_path).split('/')[-1]
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))


        images.append(img)
        labels.append(label)
        image_names.append(filename)

   


for path in glob.glob("Diseases/*"):
    label = path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        # ------------------------------ Data Augmentation-----------------------------------

        

        images.append(img)
        labels.append(label)
        image_names.append(os.path.basename(img_path).split('/')[-1])
        # ---------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------


# Conversion of image list into respective numpy array
images = np.array(images)
labels = np.array(labels)


# ------------------------------------Train Test Split------------------------------------
train_images, test_images, train_labels, test_labels, image_names_nq, image_names = train_test_split(
    images, labels, image_names, test_size=0.25, random_state=42)
# ----------------------------------------------------------------------------------------


# ------------------------------------Label Encoding------------------------------------
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
print(test_labels_encoded, test_labels)
# ---------------------------------------------------------------------------------------


# Variable updation and normalization of data
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
x_train, x_test = x_train / 255.0, x_test / 255.0


# ------------------------------------Printing the size of each data------------------------------------
print("Total Images: ", images.shape[0])
print("Training Images: ", x_train.shape[0])
print("Test Images: ", x_test.shape[0])
# ------------------------------------------------------------------------------------------------------


# ------------Function to extract features from each image and storing into a DataFrame-----------------
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):
        input_img = x_train[image, :, :, :]
        img = input_img

        df = pd.DataFrame()

        # FEATURE 1 - Bunch of Gabor filter responses

        # Generate Gabor features
        num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):  # Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.5
                # Label Gabor columns as Gabor1, Gabor2, etc.
                gabor_label = 'Gabor' + str(num)
    #                print(gabor_label)
                ksize = 9
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                # Now filter the image and add values to a new column
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                # Labels columns as Gabor1, Gabor2, etc.
                df[gabor_label] = filtered_img
                num += 1  # Increment for gabor column label
        models.append("Gabor")

        # FEATURE 2 - Laplace filter
        edge_laplace = laplace(img)
        edge_laplace1 = edge_laplace.reshape(-1)
        df['Laplace'] = edge_laplace1
        models.append("Laplace")

        # FEATURE 3 - Gaussian filter
        edge_gaussian = gaussian(
            img, sigma=1, mode='reflect', channel_axis=False)
        edge_gaussian1 = edge_gaussian.reshape(-1)
        df['Gaussian'] = edge_gaussian1

        image_dataset = image_dataset._append(df)
    return image_dataset

# ------------------------------------------------------------------------------------------------------


# ----------------------Features from training samples are being extracted & stored---------------------
print(x_train.shape)
image_features = feature_extractor(x_train)
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
print(image_features.shape)
# Reshape to #images, features
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))
print(X_for_RF.shape)
# ------------------------------------------------------------------------------------------------------


# ---------------------Training of Model using RF and fitting training data to it-----------------------
start = time()
RF_model = RandomForestClassifier(
    n_estimators=n_est, random_state=random_state, n_jobs=-1)
# Fit the model on training data
RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding


# ------------------------------------------------------------------------------------------------------


# ---------------------------------------------Predict on Test data--------------------------------------
# Extract features from test data and reshape, just like training data
print(x_test.shape)
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
print(test_features.shape)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
print(test_for_RF.shape)
# Predict on test
test_prediction = RF_model.predict(test_for_RF)
# Inverse le transform to get original label back.
test_prediction = le.inverse_transform(test_prediction)
class_data["Image"] = image_names
print(test_labels.shape, test_prediction.shape)
class_data["Actual Label"] = test_labels
class_data["Predicted Label"] = test_prediction
# class_data = class_data[class_data['Actual Label'] !=class_data["Predicted Label"]]
# print(class_data)
test_features = np.array([])
# x_test = np.array([])
# test_images = np.array([])
# ------------------------------------------------------------------------------------------------------


# -------------------------------------Predict on train for train accuracy-----------------------------
train_features = np.expand_dims(image_features, axis=0)
train_for_RF = np.reshape(train_features, (x_train.shape[0], -1))
train_prediction = RF_model.predict(train_for_RF)
# Inverse le transform to get original label back.
train_prediction = le.inverse_transform(train_prediction)

image_features = np.array([])
train_for_RF = np.array([])
train_features = np.array([])
# ------------------------------------------------------------------------------------------------------


# -----------------------Implementing cross validation------------------------
scores = cross_val_score(RF_model, X_for_RF, y_train, cv=5)
valid_accuracy = scores.mean()  # Validation Accuracy
# print("CV Score :", scores)
# print("CV Score(Mean) :", scores.mean())
print("CV Score(Standard Deviation) :", scores.std())

X_for_RF = np.array([])
# ----------------------------------------------------------------------------


# -----------------------Printing Accuracies, Scores and Classification Report--------------------------
accuracy = metrics.accuracy_score(
    y_true=test_labels, y_pred=test_prediction)  # Testing Accuracy
train_accuracy = metrics.accuracy_score(
    y_true=train_labels, y_pred=train_prediction)  # Training Accuracy
precision_score = metrics.precision_score(
    test_labels, test_prediction, average='macro')
recall_score = metrics.recall_score(
    test_labels, test_prediction, average='macro')
f1_score = metrics.f1_score(test_labels, test_prediction, average='macro')
class_report = metrics.classification_report(
    y_true=test_labels, y_pred=test_prediction)
print("Classification Report: ", class_report)
print("Testing Accuracy = ", accuracy)
print("Training Accuracy = ", train_accuracy)
print("Validation Accuracy = ", valid_accuracy)
print("Precision Score (macro)= ", precision_score)
print("Recall Score (macro)= ", recall_score)
print("F1 Score (macro)= ", f1_score)
# -------------------------------------------------------------------------------


# -------------------------------- Print confusion matrix-------------------------
cm = confusion_matrix(test_labels, test_prediction)
fig, ax = plt.subplots(figsize=(6, 6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
plt.show()
# ---------------------------------------------------------------------------------


def updateExcel():
    # --------------- Updating details in excel file of Accuracy Details and Classification Reports------------------
    df1 = pd.DataFrame({'Classifier': ['RF'],
                       'N_Estimators': [n_est],
                        'Random State': [random_state],
                        'Sample Size': [x_train.shape[0]],
                        'Accuracy': [accuracy],
                        'Training Accuracy': [train_accuracy],
                        'Validation Accuracy': [valid_accuracy],
                        'Precision Score': [precision_score],
                        'Recall Score': [recall_score],
                        'F1 Score': [f1_score],
                        'Models': [set(models)],
                        'Date and Time': [datetime.now()]
                        })
    df2 = pd.DataFrame({'Classifier': ['RF'],
                       'N_Estimators': [n_est],
                        'Random State': [random_state],
                        'Classification report': [class_report],
                        'Date and Time': [datetime.now()]
                        })
    try:
        without_aug = pd.read_excel(
            'Rice Leaf Model 2.xlsx', sheet_name="Without Augmentation")
        without_aug_CR = pd.read_excel(
            'Rice Leaf Model 2.xlsx', sheet_name="Classification Reports")
        with_aug = pd.read_excel(
            'Rice Leaf Model 2.xlsx', sheet_name="With Augmentation")
        with_aug_CR = pd.read_excel(
            'Rice Leaf Model 2.xlsx', sheet_name="Classification Reports(Aug.)")
        # read  file content
        # print(reader)
        # create writer object
        # used engine='openpyxl' because append operation is not supported by xlsxwriter
        writer = pd.ExcelWriter('Rice Leaf Model 2.xlsx', engine='openpyxl',
                                mode='a', if_sheet_exists="overlay")

        # append new dataframe to the excel sheet
        df1.style.set_properties(**{'text-align': 'center'}).to_excel(
            writer, index=False, header=False, startrow=len(with_aug) + 1, sheet_name='With Augmentation')
        df2.style.set_properties(**{'text-align': 'center'}).to_excel(
            writer, index=False, header=False, startrow=len(with_aug_CR) + 1, sheet_name='Classification Reports(Aug.)')
        # close file
        writer.close()
    except FileNotFoundError:
        writer = pd.ExcelWriter('Rice Leaf Model 2.xlsx', engine='xlsxwriter')
        df1.style.set_properties(
            **{'text-align': 'center'}).to_excel(writer, index=False, sheet_name='With Augmentation')
        df2.style.set_properties(
            **{'text-align': 'center'}).to_excel(writer, index=False, sheet_name='Classification Reports(Aug.)')

        # close file
        writer.close()
    # ------------------------------------------------------------------------------------------------------------

    # ------------------------------Updating details in excel file Prediction Analysis------------------------------
    try:
        classification = pd.read_excel(
            'Prediction Analysis 14.xlsx', sheet_name="Prediction Analysis 14")
        writer = pd.ExcelWriter('Prediction Analysis 14.xlsx', engine='openpyxl',
                                mode='a', if_sheet_exists="overlay")

        class_data.style.set_properties(**{'text-align': 'center'}).to_excel(
            writer, index=False, header=False, startrow=len(classification) + 1, sheet_name='Prediction Analysis')
    except FileNotFoundError:
        writer = pd.ExcelWriter(
            'Prediction Analysis 14.xlsx', engine='xlsxwriter')
        class_data.style.set_properties(
            **{'text-align': 'center'}).to_excel(writer, index=False, sheet_name='Prediction Analysis')

        # close file
        writer.close()
    # --------------------------------------------------------------------------------------------------------------


# updateExcel()
end = time()
print("Execution time: ", end-start)

pickle.dump(RF_model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(x_test[0].shape)
input_img = np.expand_dims(x_test[0], axis=0)
print(input_img.shape)
input_img_features = feature_extractor(input_img)
print(input_img_features.shape)
input_img_features = np.expand_dims(input_img_features, axis=0)
print(input_img_features.shape)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
print(input_img_for_RF.shape)
print(le.inverse_transform(model.predict(input_img_for_RF)))
