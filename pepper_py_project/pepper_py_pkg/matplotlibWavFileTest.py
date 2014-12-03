# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:16:52 2014

@author: boerresen
"""


from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# read audio samples
input_data = read("angry.wav")
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:1024])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot
plt.show()