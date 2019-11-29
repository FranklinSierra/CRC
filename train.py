from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from optparse import OptionParser
#para graficas estadisticas
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
#para guardar el modelo a medida que se entrene
from keras.callbacks import ModelCheckpoint
from DataSet.dataset import DataSetPolyps
from keras import optimizers



#creacion de las banderas
parser = OptionParser()
parser.add_option("--dataset", dest="dataset", type=str, 
help="Wich dataset to load", default='WL')
#new flag
parser.add_option("--model", dest="model", type=str, 
help="Wich model to load", default='VGG16')
(options, args) = parser.parse_args()
#despues del . va lo de dest en el add_option
dataset_name = options.dataset
model_name = options.model

if (str.upper(model_name)=="VGG16"):
    from Models.VGG16.vgg16Pretrained import Vgg16PreTrained
    model_class = Vgg16PreTrained()
else:
    from Models.VGG16.vgg16Pretrained import Vgg16PreTrained
    model_class = Vgg16Pretrained()

dataSet = DataSetPolyps(dataset_name)

train_df = dataSet.getTrainDataSet()
test_df = dataSet.getTestDataSet()
print(train_df)

datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

train_generator= datagen.flow_from_dataframe(dataframe=train_df,
                                             directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="training",
                                             batch_size=8,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="categorical",
                                             target_size=(112, 112))

valid_generator= datagen.flow_from_dataframe(dataframe=train_df,
                                             directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="validation",
                                             batch_size=8,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="categorical",
                                             target_size=(112, 112))

test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(dataframe=test_df,
                                                directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                                x_col="x_col",
                                                y_col=None,
                                                batch_size=8,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(112, 112))

class_names = {0:"adenoma", 1:"Hiperplastic"}
for x_batch, y_batch in train_generator:
    anImage = x_batch[0, :, : , :]
    aLabel = y_batch[0]
    aLabel = np.argmax(aLabel)
    plt.figure(figsize=(10,10))
    plt.imshow(anImage)
    plt.axis('off')
    plt.title(class_names[aLabel])
    print(len(x_batch))
    plt.show()
    break

def TrainModel(modelo, learning_rate, epochs, batch_size=8):
    #early stoping
    model = modelo.getTransferLearning()

    dataSet = DataSetPolyps(dataset_name)

    train_df = dataSet.getTrainDataSet()
    test_df = dataSet.getTestDataSet()
    print(train_df)

    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

    train_generator= datagen.flow_from_dataframe(dataframe=train_df,
                                                directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                                x_col="x_col",
                                                y_col="y_col",
                                                subset="training",
                                                batch_size=8,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="categorical",
                                                target_size=(112, 112))

    valid_generator= datagen.flow_from_dataframe(dataframe=train_df,
                                                directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                                x_col="x_col",
                                                y_col="y_col",
                                                subset="validation",
                                                batch_size=8,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="categorical",
                                                target_size=(112, 112))

    test_datagen = ImageDataGenerator(rescale=1./255.)

    test_generator=test_datagen.flow_from_dataframe(dataframe=test_df,
                                                    directory="C:\\Users\\frank\\Documents\\Tesis\\Polipos\\DataSet\\",
                                                    x_col="x_col",
                                                    y_col=None,
                                                    batch_size=8,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode=None,
                                                    target_size=(112, 112))

    #patience es # de epochs para ver mejora (puede ser %)
    #mejora superior al delta
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                    patience= int(epochs*0.05), verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    #1er par: donde guarda las graficas
    #histograma: cada cuanto se guarda los epochs
    tensor_board = TensorBoard(log_dir='./tensorBoard', histogram_freq=0, write_graph=True, write_images=False)

    #donde este el mejor acc guarda
    check_point = ModelCheckpoint('output/{val_acc:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    #coleccion de los "mejores datos"
    call_backs = [early_stopping, tensor_board, check_point]

    compileModel(model, learning_rate)

    #entrenando el modelo
    history = model.fit_generator(train_generator, steps_per_epoch= 10, epochs= epochs, verbose=1, callbacks=call_backs,
                        validation_data=valid_generator, validation_steps= 10, class_weight=None, shuffle=False, initial_epoch=0)

    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



def compileModel(model, learning_rate):
    model.compile(optimizer= optimizers.Adam(lr = learning_rate), loss='binary_crossentropy', metrics=['acc'], loss_weights=None, sample_weight_mode=None,
             weighted_metrics=None, target_tensors=None)


TrainModel(model_class, 1e-4, 100)    