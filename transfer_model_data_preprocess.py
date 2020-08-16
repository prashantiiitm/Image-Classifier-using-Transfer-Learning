# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:54:03 2020

@author: prashant.pandey
"""

import os.path
try:
    filepath = os.path.abspath(os.path.dirname(__file__))
except:
    filepath = "D:/Project Files/Personal/PizzaClassifierTransfer"
os.chdir(filepath)


import numpy as np
from sklearn.model_selection import train_test_split

import glob
from skimage import io
from skimage.transform import resize



car_images_list = glob.glob("Images\\car\\*jpg",recursive = True)
bike_images_list = glob.glob("Images\\motorbike\\*jpg",recursive = True)

img_dataset_car = []

for img_id in car_images_list:
    img = io.imread(img_id)
    
    if len(img.shape) == 3:
        resized_image = resize(img,(192,192,3),anti_aliasing = True)
        resized_image = 255 * resized_image
        # Convert to integer data type pixels.
        final_image = resized_image.astype(np.uint8)
        img_dataset_car.append(final_image)

img_dataset_bike = []

for img_id in bike_images_list:
    img = io.imread(img_id)
    
    if len(img.shape) == 3:
        resized_image = resize(img,(192,192,3),anti_aliasing = True)
        resized_image = 255 * resized_image
        # Convert to integer data type pixels.
        final_image = resized_image.astype(np.uint8)
        img_dataset_bike.append(final_image)


img_dataset_car_np = np.array(img_dataset_car)
img_dataset_bike_np = np.array(img_dataset_bike)

###########################################################

######### training and testing data 

dataset_train = np.concatenate((img_dataset_car_np[0:500],img_dataset_bike_np[0:500]),axis = 0)

label_train = np.array(np.concatenate((np.ones(500),np.zeros(500)))).astype(int)

label_train = [[x] for x in label_train]

dataset_test = np.concatenate((img_dataset_car_np[500:625],img_dataset_bike_np[500:625]),axis = 0)


############  testing data 
label_test = np.array(np.concatenate((np.ones(125),np.zeros(125)))).astype(int)

label_test = [[x] for x in label_test]

idx = np.random.permutation(len(dataset_test))

dataset_test = np.array(dataset_test)[idx]
label_test = np.array(label_test)[idx]

val_split = 0.15
X_train, X_val, y_train, y_val = train_test_split(dataset_train, label_train, test_size=val_split, stratify=label_train)


filename = 'train_test_data.pickle'
import pickle
with open(filename, 'wb') as f:
    pickle.dump([X_train,y_train,X_val,y_val,dataset_test,label_test], f)