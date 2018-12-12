#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:51:16 2018

@author: xaxi
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.recurrent import CuDNNLSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.layers import Dense, Activation, Dropout, LSTM, TimeDistributed
import math
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import maxnorm

##funcion para dividir en conjunto de entrenamiento y test.
def create_Xt_Yt(X, y, percentage=0.8):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
#    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

#data entre 0 y 1
#scaler = MinMaxScaler().fit(data[columns])
#data[columns] = scaler.transform(data[columns])


#Configuration
normalize_inputs = False
diff_inputs = True
scale_inputs = True


##parametros
Columns = ['High','Low','Close','Open'] ## the first ones correspond to the outputs.
OUT_SIZE = 1 # you select the output columns changing that.
WINDOW_LENGHT = 24
EMB_SIZE = len(Columns)
BATCH_SIZE = 64
STEP = 1
FORECAST = 1
EPOCHS = 40



###lectura de datos
original_data = pd.read_csv('BTCbitfinex.csv').get(Columns)

data = np.array(original_data)


if diff_inputs:
    data = np.array(original_data.pct_change().dropna())
    
if scale_inputs:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
#
data_lenght = data.shape[0]

num_features = data.shape[1]

num_sequences = math.floor((data_lenght)/STEP)

X = np.zeros([num_sequences-WINDOW_LENGHT,WINDOW_LENGHT,num_features])

Y = np.zeros([num_sequences-WINDOW_LENGHT,OUT_SIZE])

means,stds = np.zeros([num_sequences,num_features]), np.zeros([num_sequences,num_features])

for index,window_start in enumerate(range(0, data_lenght-WINDOW_LENGHT, STEP)):    
    try:
        
        if normalize_inputs:
            means[index] = np.mean(data[window_start:window_start+WINDOW_LENGHT],axis = 0)
            stds[index] = np.std(data[window_start:window_start+WINDOW_LENGHT],axis = 0)
            X[index] = (data[window_start:window_start+WINDOW_LENGHT]-means[index])/stds[index]
            Y[index] = (data[window_start+WINDOW_LENGHT:window_start+WINDOW_LENGHT+FORECAST][:,:OUT_SIZE]-means[index][:OUT_SIZE])/stds[index][:OUT_SIZE]
        else:
            X[index] = data[window_start:window_start+WINDOW_LENGHT]
            Y[index] = data[window_start+WINDOW_LENGHT:window_start+WINDOW_LENGHT+FORECAST][:,:OUT_SIZE]
    except Exception as e:
        print(e)
    

X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y,percentage = 0.80)

model = Sequential()
model.add(LSTM(units = 100,input_shape=(WINDOW_LENGHT,EMB_SIZE), return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units = 100, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units = 100, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units = 100, return_sequences=False))


model.add(Dense(1))


opt = Nadam(lr=0.02)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
model.compile(optimizer='adam', 
              loss='mae',
              metrics=['mae'])


history = model.fit(X_train, Y_train, 
          epochs = EPOCHS,
          batch_size = BATCH_SIZE, 
          verbose=1, 
          validation_data=(X_test, Y_test),
          callbacks=[checkpointer],
          shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#predictions = scaler.inverse_transform(model.predict(X_test))
predictions = model.predict(X_test,batch_size=BATCH_SIZE)
plt.plot(predictions[100:200])
plt.plot(Y_test[100:200])
plt.plot()


#plt.plot(model.predict(X_test)[:,0,0])
#plt.plot(Y_test[])
#plt.show()
#
#plt.plot(model.predict(X_test)[:,1])
#plt.plot(Y_test[:,1])
#plt.show()

money = 100
amount = 0
precios = np.array(original_data[-len(Y_test):])
#    for i in range(precios):
#        print(i)
#    

    
