#Preprocessing of input data samples
#The data samples is either snr(0,18,-18) based on your input

import os
import numpy as np # for array manipulation
import scipy.io
import cmath

#for loading the array data that is in format of matlab
from scipy.io import loadmat, savemat

#perfom data splitting to have train and test dataset
from sklearn.model_selection import train_test_split

#libraries for model building
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D,Flatten,LSTM,TimeDistributed
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Dropout,concatenate,AveragePooling2D


# Transmitted signal
mat2 = scipy.io.loadmat("tx.mat")
tx = mat2['tx']

U = np.zeros((4, 4), dtype=np.complex64)
for k in range(1,5):
    for l in range(1,5):     
        U[k-1,l-1] = 0.5*(np.exp(-1j*2*(np.pi)*k*l/4))
UT=np.matrix.getH(U)

#function Counting : Function to test localization model
# Where: 
#   Input:
#        rx: Input data samples
#   Output:
#        H_RAD: Preprocessed Input to be used as ML model input
def pre_processing(rx):
    
    n = 1068
    H_f = np.zeros((4,4,45,128), dtype=np.complex128)

    start = 10*128
    end = 18*128 + 45 -1
    Tx = np.transpose(tx)
    
    Y = rx[:n, :,:]
    Y2 = rx[start:end, :,:]
    Y = np.einsum('ijk->jik', Y)
    Y2 = np.einsum('ijk->jik', Y2)
  
    for i in range(128):
        # Field-1
        x_slice = np.zeros((4,1024+45-1))
        x_slice[:,:1024] = Tx[:,:1024]
        X = x_slice
        for j in range(1, 45):
            x_slice = np.roll(x_slice, 1, axis=1)
            X = np.vstack((X, x_slice))
            
        # Field-2       
        x_slice = np.zeros((4,1024+45-1))
        x_slice[:,:1024] = Tx[:,start:start+1024]
        X2 = x_slice
        for j in range(1, 45):     
            x_slice = np.roll(x_slice, 1, axis=1)
            X2 = np.vstack((X2, x_slice))

        p = np.zeros((4,1024+45-1), dtype=np.complex128)
        q = np.zeros((4,1024+45-1), dtype=np.complex128)
        p[:,:]=Y[:,:,i]
        q[:,:]=Y2[:,:,i]
        
        p_1 = np.matmul(p, np.transpose(X)) + np.matmul(q, np.transpose(X2))
        p_2 = np.matmul(X, np.transpose(X)) + np.matmul(X2, np.transpose(X2))
        k = np.linalg.inv(p_2)

        p_3=np.matmul(p_1, k)
        count=0;
        vv=np.zeros((44), dtype=np.int)
        count=0
        for kkk in range(4,180,4):
            vv[count]=kkk
            count=count+1
        p_4=np.array(np.hsplit(p_3,vv))
        p_5=np.einsum('ijk->jki', p_4)
        H_f[:,:,:,i] = p_5
        
    H_RD = np.zeros((4,4,45,16), dtype=np.complex128)
    h_slice = np.zeros((16,1), dtype=np.complex128)
    H_RAD = np.zeros((4,4,45,16), dtype=np.complex128)

    for tap in range(45):
        for i in range(4):
            for j in range(4):
                for ch in range(16):
                    h_slice[ch] = np.mean(H_f[j,i,tap,8*(ch):8*(ch+1)])
                    fftoutput=np.fft.fft(np.squeeze(h_slice))[::-1]
                    H_RD[j,i,tap,:] = np.roll(fftoutput,1)

    p_l = np.zeros((4,4), dtype=np.complex128)   
    for ch in range(16):
        for m in range(45):
            p_l[:,:]=np.matrix(H_RD[:,:,m,ch])
            p_k=np.matmul(p_l,UT)
            H_RAD[:,:,m,ch] = np.matmul(U,p_k)

    H_RAD =  abs(H_RAD)
    return H_RAD
