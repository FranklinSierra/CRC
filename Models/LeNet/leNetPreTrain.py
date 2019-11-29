from keras.layers import Dense, Dropout, Activation, Flatten, Input
import keras
import keras.layers as layers
from keras.models import Sequential

class LeNetPreTrained():
      #constructor
      def __init__(self, input_shape = (112,112,3), num_clases=2):
            self.__lenet = keras.Sequential()
            self.__lenet.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape = input_shape))
            self.__lenet.add(layers.AveragePooling2D())
            self.__lenet.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
            self.__lenet.add(layers.AveragePooling2D())
            self.__lenet.add(layers.Flatten())
            self.__lenet.add(layers.Dense(units=120, activation='relu'))
            self.__lenet.add(layers.Dense(units=84, activation='relu'))
            self.__lenet.add(layers.Dense(units=num_clases, activation = 'sigmoid'))

      
      def getFeatureExtraction(self, layer_name = 'conv2d_15'):
            capa = keras.Model(input = self.__lenet.input, outputs = self.__lenet.get_layer(layer_name).output)

                  
      def getModel(self):
            return self.__lenet