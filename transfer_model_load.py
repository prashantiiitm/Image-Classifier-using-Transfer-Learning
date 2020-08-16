# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:32:21 2020

@author: prashant.pandey
"""
######## Set up the address of the current running script
import os.path
try:
    filepath = os.path.abspath(os.path.dirname(__file__))
except:
    filepath = "D:/Project Files/Personal/PizzaClassifierTransfer"
os.chdir(filepath)

import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.applications import VGG16

from keras.optimizers import SGD, Adam

HEIGHT = 192
WIDTH = 192

base_model = VGG16(weights='imagenet' , 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))


FC_LAYERS = [1024, 1024,1024,1024 ]
dropout = 0.4

from functions import build_finetune_model



finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=2)


#
#dataset_train = dataset_train[0:100]
#label_train = label_train[0:100]


NUM_EPOCHS = 10
BATCH_SIZE = 16

adam = Adam(lr=0.000005)
finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

#history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
#                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
#                                       shuffle=True, callbacks=callbacks_list)

val_split = 0.15
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(validation_split = val_split)




import pickle
with open('train_test_data.pickle', 'rb') as f:
    X_train,y_train,X_val,y_val,dataset_test,label_test = pickle.load(f)

X = np.concatenate((X_train, X_val))
y = np.concatenate((y_train, y_val))

history = finetune_model.fit(datagen.flow(X,y,subset = 'training',batch_size = BATCH_SIZE),
                             steps_per_epoch=len(X_train) // BATCH_SIZE,epochs = NUM_EPOCHS,shuffle=True,
                             validation_data = datagen.flow(X,y,subset = 'validation',batch_size = BATCH_SIZE), validation_steps = len(X_val) // BATCH_SIZE)



testdatagen = ImageDataGenerator()
#print("Evaluate on test data")
#results = finetune_model.evaluate(testdatagen.flow(dataset_test, label_test, batch_size=BATCH_SIZE,shuffle = True))
#print("test loss, test acc:", results)

from sklearn.metrics import classification_report

y_pred = finetune_model.predict(testdatagen.flow(dataset_test, batch_size=BATCH_SIZE,shuffle = False),verbose = 1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(label_test, y_pred_bool))

import pandas as pd
out = pd.DataFrame(label_test, columns = {'actual'})
out['predicted'] = y_pred_bool
out['diff'] = abs(out['predicted'] - out['actual'])
out['diff'].sum()

sum(y_pred_bool)
