#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:28:46 2018

@author: jimmyhomefolder
"""
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import featuretools as ft
import numpy as np
startTime = datetime.now()
print(startTime)

#%%
print("File loading ... ")
dataset = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

set(dataset.Label)
features = list(set(dataset.columns) - {'Label'})
X = dataset[features] #metrics of features
Xx = test[features]
y = dataset['Label'].values # independent variable vector(outcome)

X['index'] = X.index
X['index'] = X['index'].astype(float)

entities = {
        "DLL"  : (X, "index", "DLLkaon"),
        "Flag" : (X, "index", "FlagBrem"),
        "RICH" : (X, "index", "RICH_DLLbeProton"),
        "Muon" : (X, "index", "MuonLooseFlag"),
        "Rest" : (X, "index", "BremDLLbeElectron")
        }

relationships = [("DLL", "index", "Flag", "FlagBrem"),
                  ("RICH", "index", "Muon", "MuonFlag"),
                  ("Rest", "index", "Muon", "MuonLLbeMuon")
                  ]

feature_matrix_customers, features_defs = ft.dfs(entities = entities,
                                                 relationships = relationships,
                                                 target_entity="DLL"
                                                 )   

X = pd.concat([X, feature_matrix_customers], axis=1)

#%%
print("encoding")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#%%
print("splitting data")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
print("feature Scaling")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
print("Building ANN")
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

classifier = Sequential() 
classifier.add(Dense(units = 1200, kernel_initializer = 'uniform', activation = 'relu', input_dim = 395)) 
classifier.add(Dense(units = 1200, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

print("NN start training ... ")
print(datetime.now())
#
from keras.callbacks import EarlyStopping, ModelCheckpoint
stop_here_please = [EarlyStopping(monitor='val_loss', patience = 20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

classifier.fit(X_train, 
               y_train, 
               batch_size = 40, 
               epochs = 100, 
               callbacks = stop_here_please, 
               validation_data=(X_test, y_test)
               )

#%%
print("Outputing file ...")
Xx = sc.fit_transform(Xx)
kaggle_proba = classifier.predict_proba(Xx)
kaggle_ids = test.ID 


from IPython.display import FileLink

def create_solution(ids, proba, names, filename='BaseLine_ICL2018_ANN_Keras.csv'):
    solution = pd.DataFrame({'ID': ids})
    
    for name in ['Ghost', 'Electron', 'Muon', 'Pion', 'Kaon', 'Proton']:
        solution[name] = proba[:, np.where(names == name)[0]]
    
    solution.to_csv('{}'.format(filename), index=False)
    return FileLink('{}'.format(filename))

classes = np.array(['Electron', 'Ghost', 'Kaon', 'Muon', 'Pion', 'Proton'])
create_solution(kaggle_ids, kaggle_proba, classes)

print("Done.")
print("Time it takes to finish:")
print(datetime.now() - startTime)