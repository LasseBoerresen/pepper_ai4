import math
import numpy as np
import scipy as sp
from scipy import ndimage, misc
import random
import skimage
from skimage import data, filter, io
import matplotlib.pyplot as plt

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

net = FeedForwardNetwork()
inLayer = LinearLayer(4)
hiddenLayer = SigmoidLayer(2)
outLayer = LinearLayer(4)

net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()

net.params