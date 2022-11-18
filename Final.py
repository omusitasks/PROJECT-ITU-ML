import os
from scipy.io import loadmat, savemat
import numpy as np
import scipy.io
import PreProcessing
import NN_model
import Counting
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

#INPUT PARAMETERS
#snr = ###### INPUT SNR HERE ######


#Instruction 1#
### Please Run this code where snr18, snr0 and snr-18 folders are placed.###
###Input####
print("Enter the Parent folder path:")
ParentFilePath = input() 
print("Enter the snr value for testing:")
snr = int(input())
filepath = 'snr'
#Function loadfile : Loads file for snr 18, 0 and -18
# Where:
#      snr = SNR 18,0, -18
#      num= sample number  
def loadfile(snr, num):
    rx = loadmat(ParentFilePath+filepath + str(snr) + '/rxSigCh' + str(num) + '.mat')['rxNoisy']
    return rx


#function Testing : Function to test localization model
# Where:
#    Input : 
#       Y_Pred: prediction vector (softmax output for all the sectors for different number of people) from the ML model 
#       num_labels: maximum number of people assumed in a sector for ml model training : 4 classes : either 0 person or 1  or 2 or 3 persons 
#   Output:
#        Sector Num: number of people present in each sector for each sample
def Testing(Y_pred, num_labels):
    
    # People count in each sector of the grid
    SectorNum = np.zeros((NumTestSamples, 9))
    for i in range(NumTestSamples):
        for j in range(9):
            SectorNum[i][j]=(np.argmax(Y_pred[i][j*num_labels:(j+1)*num_labels]))
    
    return SectorNum


#function sector_form : Funtion to convert model predictions into sector format string
# Where:
#    Input :
#       h_rad: data samples after preprocessing
#       Y_Pred: prediction vector (softmax output for all the sectors for different number of people) from the ML model 
#       Y_pred_label: prediction class labels from ML model for localization
#       SNR: snr value
#   Output: It creates a .txt file to store sector format strings
def sector_form(h_rad, Y_pred, Y_pred_label, SNR):
    
    num_test = np.shape(Y_pred)[0]
    len_predict = np.zeros((num_test, 1))
    ind = np.zeros((num_test, 18), dtype=int)
    diff = np.zeros((num_test, 9), dtype=float)
    ind_sort = np.zeros((num_test, 9), dtype=int)
    
    len_pred = model_count.predict([h_rad])
    
    for i in range(num_test):
        a = int(np.argmax(len_pred[i,:]))
        len_predict[i] = a + 1
    print(len_predict.shape)
    
    for i in range(num_test):
        for j in range(9):
            ind[i][2*j:2*(j+1)] = np.argpartition(Y_pred[i][j*4:(j+1)*4], -2)[-2:]
            diff[i][j] = abs(Y_pred[i][ind[i][2*j]] - Y_pred[i][ind[i][2*j + 1]])
        ind_sort[i][:] = np.argsort(diff[i][:])
    
    for i in range(num_test):
        lin = ""
        for j in range(9):
            num_people = int(np.argmax(Y_pred[i][j*4:(j+1)*4]))
            c = chr(ord('A') + j)
            for k in range(num_people):
                lin = lin + c

        p = 0
        if(len(lin) < len_predict[i]):
            aa = len_predict[i] - len(lin)
            for mn in range(9):
                if(aa==0):
                    break
                Sno = ind_sort[i,mn]
                curr_sector = chr(ord('A') + Sno)
                if(Y_pred[i][ind[i][2*Sno]] >= Y_pred[i][ind[i][2*Sno + 1]]):
                    if(ind[i][2*Sno+1] > ind[i][2*Sno]):
                        p = ind[i][2*Sno+1] - ind[i][2*Sno]      
                else:
                    if(ind[i][2*Sno+1] < ind[i][2*Sno]):
                        p = ind[i][2*Sno] - ind[i][2*Sno+1]
                v = int(np.min((aa,p)))
                for kk in range(v):
                    lin = lin + curr_sector
                    aa = aa - 1

            for k in range(int(aa)):
                lin = lin + 'A'

        elif(len(lin) > len_predict[i]):
            aa = len(lin) - len_predict[i]
            for mn in range(9):
                if(aa==0):
                    break
                Sno = ind_sort[i,mn]
                curr_sector = chr(ord('A') + Sno)
                if(Y_pred[i][ind[i][2*Sno]] >= Y_pred[i][ind[i][2*Sno + 1]]):
                    if(ind[i][2*Sno+1] < ind[i][2*Sno]):
                        p = ind[i][2*Sno] - ind[i][2*Sno+1]      
                else:
                    if(ind[i][2*Sno+1] > ind[i][2*Sno]):
                        p = ind[i][2*Sno+1] - ind[i][2*Sno]
                v = int(np.min((aa,p)))
                for kk in range(v):
                    ii = lin.find(curr_sector)
                    if(ii == -1):
                        break
                    lin_1 = lin[:ii]
                    lin_2 = lin[(ii+1):]
                    lin = lin_1 + lin_2
                    aa = aa - 1

            for k in range(int(aa)):
                lin = lin[:-1]
        f = open('mlResult'+ str(SNR) + '.txt', 'a')
        f.write(lin)
        f.write("\n")
        f.close()
        

# Main function
NumTestSamples = np.shape(os.listdir(filepath + str(snr)))[0]
Y_pred = np.zeros((NumTestSamples, 9), dtype=float)
h_rad = np.zeros((NumTestSamples,4,4,45,16), dtype=float)

#Data Preprocessing
for i in range(NumTestSamples):
    rx = loadfile(snr, i)
    h_rad[i,:,:,:,:] = PreProcessing.pre_processing(rx) 

#ML model predictions for people count
model_count = Counting.CountingPeople()

#Loading ML model for localization
model = NN_model.load_NNweights()
#Prediction on test data
Y_pred = model.predict([h_rad])
#Conversion of predictions to strings in sector format
Y_pred_sector = Testing(Y_pred, 4)
sector_form(h_rad, Y_pred, Y_pred_sector, snr)
