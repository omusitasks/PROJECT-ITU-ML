#ML model for counting number of people present in the complete grid

import os
from scipy.io import loadmat, savemat
import numpy as np
import scipy.io
import cmath
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D,Flatten,LSTM,TimeDistributed
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Dropout,concatenate,AveragePooling2D


#function Counting : Function to test localization model
# Where: 
#   Output:
#        model: Trained weights for localization ML model
def CountingPeople():
    
    input1 = Input(shape=(4,4,45,16))

    x=Flatten()(input1)
    x=BatchNormalization()(x)
    x=Dense(2000, activation='relu')(x)
    x=Dense(500,activation='relu')(x)
    output=Dense(8,activation='softmax')(x)

    model = Model(inputs=input1, outputs=output)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.00005, decay_steps=10000, decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    
    #Loading training weights
    model.load_weights('model_018m18_count_e80.h5')
    
    return model