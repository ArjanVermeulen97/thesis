# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:56:10 2021

@author: Arjan
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

categorical = True

scaler = MinMaxScaler()

df_test = pd.read_csv('data_test.csv')
df_val = pd.read_csv('data_val.csv')
df_data = pd.read_csv('data_train.csv')

y_train = df_data.pop('MOID')
x_train = scaler.fit_transform(df_data)
y_test = df_test.pop('MOID')
x_test = scaler.transform(df_test)
y_val = df_val.pop('MOID')
x_val = scaler.transform(df_val)


if categorical:
    y_train = pd.DataFrame([1 if item <= 0.1 else 0 for item in y_train])
    y_test = pd.DataFrame([1 if item <= 0.1 else 0 for item in y_test])
    y_val = pd.DataFrame([1 if item <= 0.1 else 0 for item in y_val])
    loss = keras.losses.BinaryCrossentropy()
    lastlayer = 'sigmoid'
    print('Class balance:')
    print(y_train.value_counts())
else:
    loss = 'mse'
    lastlayer='relu'

model = keras.Sequential(
    [
     layers.Dense(x_train.shape[1], activation='tanh'),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.5),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.5),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.2),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.2),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.2),
     layers.Dense(4096, activation='relu'),
     layers.BatchNormalization(),
     layers.Dropout(0.2),
     layers.Dense(1, activation=lastlayer)
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    metrics=[keras.metrics.Precision(), 
             keras.metrics.Recall()],
    loss=loss
    )


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-15)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=50,
                           restore_best_weights=True)

history = model.fit(
            x_train,
            y_train,
            batch_size=1024,
            epochs=512,
            validation_data=(x_val, y_val),
            callbacks=[reduce_lr, early_stop]
            )

model.save('model_dense')

hist = history.history
plt.subplot(131)
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.subplot(132)
plt.plot(hist['precision'])
plt.plot(hist['val_precision'])
plt.subplot(133)
plt.plot(hist['recall'])
plt.plot(hist['val_recall'])