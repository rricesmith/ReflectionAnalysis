import os
import pickle
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
import random
import math
from datetime import date

import matplotlib
matplotlib.use('Agg')

#Set parameters
path = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
model_path = f'DeepLearning/models/'                                                #Path to save models
plots_path = 'DeepLearning/plots/'                                                  #Path to save plots
amp = '100s'                                                                        #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
TrainCut = 5000                                                                     #Number of events to use for training

#Load RCR data
RCR_files = []
print(f'path {path + RCR_path}')
for filename in os.listdir(path + RCR_path):
    print(f'filename {filename}')
    if filename.startswith(f'FilteredSimRCR_{amp}_'):
        print(f'appending')
        RCR_files.append(path + RCR_path +  filename)
RCR = np.empty((0, 4, 256))
print(f'rcr files {RCR_files}')
for file in RCR_files:
    print(f'RCR file {file}')
    RCR_data = np.load(file)[0:, 0:4]
    print(f'RCR data shape {RCR_data.shape} and RCR shape {RCR.shape}')
    RCR = np.concatenate((RCR, RCR_data))

#Load Backlobe data
Backlobes_files = []
for filename in os.listdir(path + backlobe_path):
    if filename.startswith(f'Backlobe_{amp}_'):
        Backlobes_files.append(path + backlobe_path + filename)
Backlobe = np.empty((0, 4, 256))
for file in Backlobes_files:
    print(f'Backlobe file {file}')
    Backlobe_data = np.load(file)[0:, 0:4]
    Backlobe = np.concatenate((Backlobe, Backlobe_data))


#Make a cut on data, and then can run model on the uncut data after training to see effectiveness
#Ie train on 5k, test on 3k if total is 8k
#take a random selection because events are ordered based off CR simulated, so avoids overrepresenting particular Cosmic Rays
RCR = RCR[np.random.choice(RCR.shape[0], size=TrainCut, replace=False), :]
Backlobe = Backlobe[np.random.choice(Backlobe.shape[0], size=TrainCut, replace=False), :]

print(f'Shape RCR {RCR.shape} Shape Backlobe {Backlobe.shape}')

x = np.vstack((RCR, Backlobe))
  
n_samples = x.shape[2]
n_channels = x.shape[1]
x = np.expand_dims(x, axis=-1)
#Zeros are Backlobe, 1 signal
#y is output array
y = np.vstack((np.zeros((RCR.shape[0], 1)), np.ones((Backlobe.shape[0], 1))))
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]
print(x.shape)

BATCH_SIZE = 32 #Iterate over many epochs to see which has lowest loss
EPOCHS = 100 #Then change epochs to be at lowest for final result

#This automatically saves when loss increases over a number of patience cycles
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
def training(j):
    model = Sequential()
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1)))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    # Save the history as a pickle file
    with open(f'{model_path}/{str(date.today())}_RCR_Backlobe_model_2Layer_{j}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model {j+1}: Simulation File Used {simulation_multiplier} Times')

    # Save the loss plot as an image file
    plt.savefig(f'{plots_path}/loss_plot_{str(date.today())}_RCR_Backlobe_model_2Layer_{j}.png')
    plt.clf()

    # Plot the training and validation accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model {j+1}: Simulation File Used {simulation_multiplier} Times')

    # Save the accuracy plot as an image file
    plt.savefig(f'{plots_path}/plots/accuracy_plot_{str(date.today())}_RCR_Backlobe_model_2Layer_{j}.png')
    plt.clf()

    model.summary()

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')

    #input the path and file you'd like to save the model as (in h5 format)
    model.save(f'{model_path}/{str(date.today())}_RCR_Backlobe_model_2Layer_{j}.h5')
  
#can increase the loop for more trainings is you want to see variation
for j in range(1):
  training(j)