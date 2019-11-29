from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model
import keras

class MobilNetPreTrained():
      #constructor
      def __init__(self, input_shape =(112, 112, 3), num_clases):
            mobil = MobileNet(weights = "imagenet", include_top= False, input_shape= input_shape)
            #almaceno la salida
            x = mobil.output
            #aplico transferlearning
            for layer in mobil.layer:
                  layer.trainable = False
                        
            x = Dense(num_clases, activation='sigmoid', name = 'predictions')(mobil.layers[-2].output)
            #creo el modelo
            self.__my_mobil = Model(input =mobil.input, output = x)
            
      
      def getTransferLearning(self):
            return self.__my_mobil

      def getFeatureExtraction(self, layer_name='dense_2'):
            #Feature extraction
            capa = keras.Model(inputs= self.__my_mobil.input, outputs = self.__my_mobil.get_layer(layer_name).output)
            return capa