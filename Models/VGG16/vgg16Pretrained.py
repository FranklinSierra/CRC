from tensorflow import keras
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
#from keras.models import model_from_json
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

class Vgg16PreTrained():
    #constructor valores por defecto se dejan de ultimo (como num_clases)!!!!
    def __init__(self, input_shape = (112, 112, 3), num_clases=2):
        modelo_vgg16 = VGG16(weights="imagenet", include_top=False, input_shape = input_shape)
        #Obtengo las caracteristicas previas a la full conected a partir de la entrada 
        x=modelo_vgg16.output
        #aplico transfer learning
        for layer in modelo_vgg16.layers:
            layer.trainable = False
        #Adiciono las capas full conected
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(num_clases, activation='sigmoid', name='predictions')(x)
        #Creating my own model
        self.__my_vgg16 = Model(input = modelo_vgg16.input, output = x)
        
        for layer in self.__my_vgg16.layers:
            print(layer.name, layer.trainable)

    def getTransferLearning(self):
        return self.__my_vgg16

    def getFeatureExtraction(self, layer_name="block1_conv1"):
        #Feature extraction
        capa = keras.Model(inputs= self.__my_vgg16.input, outputs = self.__my_vgg16.get_layer(layer_name).output)
        return capa