from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1./255.)

#Para WL

train_wl_df = pd.read_csv('trainWl.csv')
columns = ['x_col', 'y_col']
train_wl_df.columns = columns

test_wl_df = pd.read_csv('testWl.csv')
columns = ['x_col', 'y_col']
test_wl_df.columns = columns


train_generator2= datagen.flow_from_dataframe(dataframe=train_wl_df,
                                             directory=None,
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="training",
                                             batch_size=32,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="binary",
                                             target_size=(576, 768))


valid_generator2= datagen.flow_from_dataframe(dataframe=train_wl_df,
                                             directory=None,
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="validation",
                                             batch_size=32,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="binary",
                                             target_size=(576, 768))
                                        
test_generator2 = test_datagen.flow_from_dataframe(dataframe=test_wl_df,
                                                directory=None,
                                                x_col="x_col",
                                                y_col=None,
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(576, 768))

class_names = {0:"adenoma", 1:"Hiperplastic"}
for x_batch, y_batch in train_generator2:
    anImage = x_batch[0, :, : , :]
    #anImage = np.array(anImage, np.uint8)
    aLabel = y_batch[0]
    plt.figure(figsize=(10,10))
    plt.imshow(anImage)
    plt.axis('off')
    plt.title(class_names[aLabel])
    print(len(x_batch))
    #print(x_batch.shape)
    plt.show()
    break

#Para NBI

train_nbi_df = pd.read_csv('trainNbi.csv')
columns = ['x_col', 'y_col']
train_nbi_df.columns = columns

test_nbi_df = pd.read_csv('testNbi.csv')
columns = ['x_col', 'y_col']
test_nbi_df.columns = columns


train_generator= datagen.flow_from_dataframe(dataframe=train_nbi_df,
                                             directory=None,
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="training",
                                             batch_size=32,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="binary",
                                             target_size=(576, 768))

valid_generator= datagen.flow_from_dataframe(dataframe=train_nbi_df,
                                             directory=None,
                                             x_col="x_col",
                                             y_col="y_col",
                                             subset="validation",
                                             batch_size=32,
                                             seed=42,
                                             shuffle=True,
                                             class_mode="binary",
                                             target_size=(576, 768))

test_generator=test_datagen.flow_from_dataframe(dataframe=test_nbi_df,
                                                directory=None,
                                                x_col="x_col",
                                                y_col=None,
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(576, 768))
                                            
class_names = {0:"adenoma", 1:"Hiperplastic"}
for x_batch, y_batch in train_generator:
    anImage = x_batch[0, :, : , :]
    #anImage = np.array(anImage, np.uint8)
    aLabel = y_batch[0]
    plt.figure(figsize=(10,10))
    plt.imshow(anImage)
    plt.axis('off')
    plt.title(class_names[aLabel])
    print(len(x_batch))
    plt.show()
    #print(x_batch.shape)
    break