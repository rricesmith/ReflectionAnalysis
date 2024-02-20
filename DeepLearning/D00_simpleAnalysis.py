import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft



#This code performs a simple analysis of data from backlobe & reflected cosmic ray events

series = '100s' #indicate which series this will be performed on
folder = 'DeepLearning/data/4thpass/'

#import the numpy arrays for the backlobe and reflected events
for file in os.listdir(folder):
    if file.startswith(f'Backlobe_{series}'):
        backlobe_traces = np.load(folder + file)
    if file.startswith(f'SimParams_Backlobe_{series}'):
        backlobe_params = np.load(folder + file)
    if file.startswith(f'SimWeights_Backlobe_{series}'):
        backlobe_weights = np.load(folder + file)
    if file.startswith(f'FilteredSimRCR_{series}'):
        reflected_traces = np.load(folder + file)
    if file.startswith(f'SimParams_SimRCR_{series}'):
        reflected_params = np.load(folder + file)
    if file.startswith(f'SimWeights_SimRCR_{series}'):
        reflected_weights = np.load(folder + file)

#Confirm the shape of the files.  The first dimension is the number of events, the second is the number of antennas, and the third is the number of time samples
#The number of events are different, but the number of antennas and time samples are the same, 4 and 256 respectively
print(f'backlobe traces shape {backlobe_traces.shape}')
print(f'reflected traces shape {reflected_traces.shape}')

#Plot the first 10 events of the backlobe and reflected events
for i in range(10):
    plt.plot(backlobe_traces[i,0,:], label=f'backlobe event {i}')
    plt.plot(reflected_traces[i,0,:], label=f'reflected event {i}')
plt.title('Backlobe Events')
plt.legend()
plt.show()
plt.clf()

#Confirm the shape of the weights file. There is only one dimension, the number of events, which is the same as the number of events in the traces files and contains the weight of the event
#The weight is the event rate of the event, based off the simulation energy and zenith, in units Evts/station/second. Higher event rate means it is a more probable event to get
print(f'backlobe weights shape {backlobe_weights.shape}')
print(f'reflected weights shape {reflected_weights.shape}')

#Confirm the shape of the parameters file. The first dimension is the number of events, and the second dimension is the number of parameters
#There are 3 parameters, the energy, zenith, and azimuth of the cosmic ray event that caused the radio signal
print(f'backlobe params shape {backlobe_params.shape}')
print(f'reflected params shape {reflected_params.shape}')

#Make a radial plot of the first 10 events, with azimuth being the angle, zenith the radius, and energy the heat of the event
#Energy is in units log10 eV, zenith is in units degrees, and azimuth is in units degrees
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
colors = reflected_params[:10, 0]  # Get the energy values for the first 10 events
ax.scatter(np.deg2rad(reflected_params[:10, 2]), np.deg2rad(reflected_params[:10, 1]), c=colors, cmap='viridis', label='Reflected Events')
#Do the same for backlobe events
colors = backlobe_params[:10, 0]  # Get the energy values for the first 10 events
sc = ax.scatter(np.deg2rad(backlobe_params[:10, 2]), np.deg2rad(backlobe_params[:10, 1]), c=colors, cmap='viridis', label='Backlobe Events')
plt.legend()
plt.colorbar(sc, label='Energy (log10 eV)')
plt.title('Radial Plot of Reflected Events')
plt.show()
plt.clf()


#Before we start plotting traces and frequencies, its good to get our x-axis for the traces and frequencies
#these don't change, so just need to do it once. It is based off of our sampling frequency for the detector, which is 2GHz, or 2 samples per nanosecond
sampling_rate = 2
x = np.linspace(0, int(256 / sampling_rate), num=256)
x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

#Now we will do a simple analysis of the data
#First we will look at the average trace of a backlobe and reflected event
average_backlobe = np.mean(backlobe_traces, axis=0)
average_reflected = np.mean(reflected_traces, axis=0)
fig, ax = plt.subplots(4, 1)
for i in range(4):
    ax[i].plot(x, average_backlobe[i], label='Average Backlobe')
    ax[i].plot(x, average_reflected[i], label='Average Reflected')
    ax[i].legend()
plt.title('Average Traces')
plt.legend()
plt.show()
plt.clf()

#Now we can compare that to the weighted average of the traces
#This is different because higher energy events, which have larger deviation, have lower weight so they don't affect the average as much
weighted_average_backlobe = np.average(backlobe_traces, axis=0, weights=backlobe_weights)
weighted_average_reflected = np.average(reflected_traces, axis=0, weights=reflected_weights)
fig, ax = plt.subplots(4, 1)
for i in range(4):
    ax[i].plot(x, weighted_average_backlobe[i], label='Weighted Average Backlobe')
    ax[i].plot(x, weighted_average_reflected[i], label='Weighted Average Reflected')
    ax[i].legend()
plt.title('Weighted Average Traces')
plt.legend()
plt.show()
plt.clf()

#We can do the same with the FFT of the traces. This is just the frequency spectrum of each trace
#In order to do this, we need the sampling rate of our detector. This is 2 GHz = 2 samples per nanosecond
backlobe_ffts = np.abs(fft.time2freq(backlobe_traces, sampling_rate))
reflected_ffts = np.abs(fft.time2freq(reflected_traces, sampling_rate))
fig, ax = plt.subplots(4, 1)
for i in range(4):
    ax[i].plot(x_freq, np.mean(backlobe_ffts[i], axis=0), label='Average Backlobe')
    ax[i].plot(x_freq, np.mean(reflected_ffts[i], axis=0), label='Average Reflected')
    ax[i].legend()
plt.title('Average FFTs')
plt.legend()
plt.show()
plt.clf()


#Now we can plot the average FFT of the backlobe and reflected events



#Then we can look at the weighted average FFT


#And finally, what should be done is the weighted average of traces and FFTs, but also average each channel
#So we want a single plot, on the left is the weighted average of all channels for reflected and backlobe events
#And on the right is the weighted average of the FFTs of all channels for reflected and backlobe events
total_weighted_average_backlobe = []
total_weighted_average_reflected = []
total_weighted_average_backlobe_fft = []
total_weighted_average_reflected_fft = []

fig, ax = plt.subplots(1, 2, sharex=False)
ax[0].plot(x, total_weighted_average_backlobe, label='Weighted Average Backlobe')
ax[0].plot(x, total_weighted_average_reflected, label='Weighted Average Reflected')
ax[0].legend()
ax[1].plot(x_freq, total_weighted_average_backlobe_fft, label='Weighted Average Backlobe')
ax[1].plot(x_freq, total_weighted_average_reflected_fft, label='Weighted Average Reflected')
ax[1].legend()
plt.title('Total Weighted Average Traces and FFTs')
plt.show()
plt.clf()



#Some next steps that can be done:
#   look at distribution of peaks in the FFTs
#   This can be done with a function like scipy.signal.find_peaks : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
#   We expect that backlobe events will have a good distribution of peaks, while reflected events will have a single peak

#Do work here
peaks_backlobe = []
peaks_reflected = []
plt.hist(peaks_backlobe, bins=100, range=(x_freq[0], x_freq[-1]), alpha=0.5, label='Backlobe')
plt.hist(peaks_reflected, bins=100, range=(x_freq[0], x_freq[-1]), alpha=0.5, label='Reflected')
plt.title('Distribution of Peaks in FFTs')
plt.legend()
plt.show()


