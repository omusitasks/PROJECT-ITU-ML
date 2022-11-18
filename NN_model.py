#ML model for localization

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D,Flatten,LSTM,TimeDistributed
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Dropout,concatenate,AveragePooling2D

#function Counting : Function to test localization model
# Where: 
#   Output:
#        model: Trained weights for people counting ML model
def load_NNweights():
    
    # Input to the model
    input1 = Input(shape=(4,4,45,16))
    
    # Neural Network model used for training and testing purpose
    x=Flatten()(input1)
    x=BatchNormalization()(x)
    x=Dense(2000, activation='relu')(x)
    x=Dense(500,activation='relu')(x)
    output = Dense(4, activation='softmax')(x)
    for i in range(8):
        a = Dense(4, activation='softmax')(x)
        output=concatenate([output,a])

    model = Model(inputs=input1, outputs=output)
    
    #Model parameters
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005,
        decay_steps=10000,
        decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    #Model compilation
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    
    #Loading training weights
    model.load_weights('model_018m18_4class_0909_ep200b.h5')
    
    return model