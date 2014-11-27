import os
import math
import time
import numpy as np
import scipy as sp
from scipy import ndimage, misc
import random
import skimage
from skimage import data, filter, io
import matplotlib.pyplot as plt

import pickle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
#from pybrain.tools.xml.networkwriter import NetworkWriter
#from pybrain.tools.xml.networkreader import NetworkReader


#shallow sparse autoencoder, with inputlayer, hiddenLayer, and output. 
#Uses backpropagatimport matplotlib.pyplot as pltion to train identity function for input, given a set of 8x8 images sampled from larger images. .
# a = w*x
# o = sigmoid(a)
#
class autoEncoder:
    def __init__(self):
        self.learningrate = 0.00001        
        self.initEpsilon = 0.01        
        self.numHidden = 1        
        self.nn = [64,32,64]
        self.B = [ (np.random.randn(1,self.nn[1]))*self.initEpsilon, (np.random.randn(1,self.nn[2]))*self.initEpsilon ]#waits of bias = 1 to every note in every layer goes         
        self.W = [ (np.random.randn(self.nn[0],self.nn[1]))*self.initEpsilon, (np.random.randn(self.nn[1],self.nn[2]))*self.initEpsilon ] #list of matrices mapping from one layer to another, fx input_64 to hidden_32 -> 1x64*64*32 = 1*32 
                
        self.Z = []        
        self.A = []
        self.E = []


    def train(self,traningSet):       
        for i in range(trainingSet.shape[1]):
            #train with image i (i.e. column i of trainingSet)            
            #First calculate activations of each layer   
            self.feedForward(trainingSet.T[i])
            self.backProbagation()
            
    
    def feedForward(self,x):
        #iterate feed forward over each layer
        #Z is a list of the linear activations of each layer.        
        self.Z = []
        self.A = []
                
        #input is seen as the activation of the 0th layer        
        self.A.append(x)
        
        #calculate activations for each layer
        for i in range(self.numHidden + 1):
            #calculate raw activations
            self.Z.append( (self.A[i].dot( self.W[i] ))+B[i] )#1x64 dot 64x32 + 1x32-> 1x32 
            #initialize activation list
            a = []           
            #calculate sigmoid for every entry in the latest entry in Z
            for j in range(self.nn[i+1]): 
                a.append(1.0/(1.0+math.exp(-self.Z[i][j])))
            #save  the activations
            self.A.append(np.array(a))
            

    def backPropagation(self):
        self.E = []
        #firstly, calculate error for the output layer       
        e = []
        #for all elements of output layer, calculate error derivative, e = delta.
        for j in range(self.nn[-1]):
            #error for each output node given by: e^(n_l) = <(y-a^(l)),f'(z^(n_l)>, with f'() given by the derivative of the sigmoid function: f'(z) a^(l)*(1-a^(l))
            e.append(-(self.A[0][j]-self.A[-1])*(self.A[-1][j]*(1-self.A[-1][j])) )
        
        self.E.append(e)
        
        #use output layer error for consecutive errorprobagation layer for layer
        for i in range(self.numHidden):
            e = []            
            
            self.E.append(np.array(e))










class imageCleaver:
    def __init__(self):
        self.numImages = 1
        self.numPatches = 5000
        self.sizePatches = 8#8 must be equal number, to be able to split in half later
        self.imageDataBase = [misc.lena()]
        self.patchDataBase = []
        self.concImgArray = np.zeros((self.sizePatches*self.sizePatches,self.numPatches))
        for i in range(self.numPatches):
             #should be random num
            randPosX = random.randint(self.sizePatches/2, self.imageDataBase[self.numImages-1].shape[0]-self.sizePatches/2)            
            randPosY = random.randint(self.sizePatches/2, self.imageDataBase[self.numImages-1].shape[1]-self.sizePatches/2)                        
            self.patchDataBase.append(self.imageDataBase[self.numImages-1][randPosX-self.sizePatches/2:randPosX+self.sizePatches/2,randPosY-self.sizePatches/2:randPosY+self.sizePatches/2])
        self.concatenateImages()
#        print(patchDataBase)
#        plt.imshow(patchDataBase[0])    
        

    #to use the images most easily for inputs to a NN, they are unravelled from 8x8 to 64x1,
    #and then concatenated so each column is another image.        
    def concatenateImages(self):        
#        imgArray = np.array([[]])
        concImgArrayT = np.transpose(self.concImgArray)
        
        for i in range(self.numPatches):     
            concImgArrayT[i] = self.patchDataBase[i].ravel()
        self.concImgArray = np.transpose(concImgArrayT)
#        print(self.concImgArray.shape)
#        plt.imshow(self.patchDataBase[0],cmap=plt.cm.gray)
#        plt.imshow(self.imageDataBase[0],cmap=plt.cm.gray)
#        print(concImgArrayT)
#            for j in range(self.sizePatches):
#                for k in range(self.sizePatches):
#                                        
#                    self.concImgArray[i] = 

    def getImages(self):    
        return 3
        
#    def showImageGrid(self):
#       #http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
#       fig, axes = plt.subplots(3, 6, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []}) 
#       fig.subplots_adjust(hspace=0.3, wspace=0.05)
#       for ax, interp_method in zip(axes.flat, methods):
#           ax.imshow(grid, interpolation=interp_method)
#           ax.set_title(interp_method)
#
#        plt.show()
#        return 1

def printNetwork(net):
    for mod in net.modules:
      print "Module:", mod.name
      if mod.paramdim > 0:
        print "--parameters:", mod.params
      for conn in net.connections[mod]:
        print "-connection to", conn.outmod.name
        if conn.paramdim > 0:
           print "- parameters", conn.params
      if hasattr(net, "recurrentConns"):
        print "Recurrent connections"
        for conn in net.recurrentConns:             
           print "-", conn.inmod.name, " to", conn.outmod.name
           if conn.paramdim > 0:
              print "- parameters", conn.params

#def saveNetParamsToFile(net):
#    f = open('netParams.txt', 'w')
#    for i in range(len(net.params)):
#        f.write(str(net.params[i]))
#        f.write('\n')
#    f.close()
#
#def loadNetParamsFromFile(net):
#    f = open('netParams.txt', 'r')
#    paramsArr = []    
#    for i in range(len(net.params)):
#        float(f.readline())
#        paramsArr.append(0.0)#(float(f.readline()))
#    net.__setattr__('params', paramsArr)
#    print net.params
#    f.close()




def main():
    #sample patches from image database
    myCleaver = imageCleaver()
    
    #instantiate NN with size as defined in image cleaver class. 
    #TODO: size should be input to imageCleaver class.
#    net = buildNetwork(myCleaver.sizePatches*myCleaver.sizePatches, (myCleaver.sizePatches*myCleaver.sizePatches)/2.0, myCleaver.sizePatches*myCleaver.sizePatches, bias = True)
    net = FeedForwardNetwork()
    inLayer = LinearLayer(myCleaver.sizePatches*myCleaver.sizePatches)
    hiddenLayer = SigmoidLayer((myCleaver.sizePatches*myCleaver.sizePatches)/4.0)
    outLayer = LinearLayer(myCleaver.sizePatches*myCleaver.sizePatches)
    
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)
    
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    net.sortModules()

#    fileObject = open('pickledNet.dat','r')
#    net = pickle.load(fileObject)
#    fileObject.close()
#    net = NetworkReader.readFrom('filename.xml') 

#    print(net.activate([2, 1]))
    #Put imageCleaver dataset into pyBrain dataset format.
    ds = SupervisedDataSet(myCleaver.sizePatches*myCleaver.sizePatches, myCleaver.sizePatches*myCleaver.sizePatches)
    for i in range(myCleaver.concImgArray.shape[1]):
        ds.addSample(myCleaver.concImgArray.T[i]/256.0,myCleaver.concImgArray.T[i]/256.0)
    
#    for inpt, target in ds:
#        print inpt, target

    
    
    trainer = BackpropTrainer(net, ds)    
    for i in range(10):    
        print(trainer.train())


    fileObject = open('./pickledNet.dat', 'w')
    pickle.dump(net, fileObject)
    fileObject.close()
#    NetworkWriter.writeToFile(net, 'testNetwork8.xml')   
        
#    saveNetParamsToFile(net)  
#    loadNetParamsFromFile(net)    
    
    imitationActivations = net.activate(myCleaver.concImgArray.T[0]/256.0)
    imitation = np.reshape(imitationActivations,(myCleaver.sizePatches,myCleaver.sizePatches))
    

    plt.figure(1)
    plt.title('Input vs output')
    
    plt.subplot(221)
    plt.imshow(myCleaver.patchDataBase[0],cmap=plt.cm.gray, interpolation='nearest',  vmin=0, vmax=256)
    plt.title('input')
        
    plt.subplot(223)
    plt.imshow(imitation*256,cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=256)
    plt.title('imitation')    
    
##    plt.show()
#    print 'imitation'
#    print imitation*256    
    

    imitationActivations2 = net.activate(myCleaver.concImgArray.T[1]/256.0)
    imitation2 = np.reshape(imitationActivations2,(myCleaver.sizePatches,myCleaver.sizePatches))
    
#    plt.figure(2)
#    plt.title('Input vs output2')
    
    plt.subplot(222)
    plt.imshow(myCleaver.patchDataBase[1],cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=256)
    plt.title('input2')

    
    plt.subplot(224)
    plt.imshow(imitation2*256,cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=256)
    plt.title('imitation2')    
    
##    plt.subplot_tool()
#    plt.show()
#    print 'imitation2'
#    print imitation2*256#   




    #################################################################
    #calculate and show each hidden nodes learned function, i.e. which input vector maximally excites the hidden node, with constrain ||x||^2 <=1
    #################################################################
    
    learned = []
    
    #convert dims from float to integer    
    i2hid = int(in_to_hidden.indim)
    i2hod = int(in_to_hidden.outdim)
    print i2hid
    print i2hod
    #fo through each hidden node
    for i in range(i2hod):
            
        one_learned  = []         
        sumOfWeights = 0
        #go through weights for the ith hidden node, sum weights        
        for j in range(i2hid):
                #add to sum, the value of connection between input                
                sumOfWeights += (in_to_hidden.params[i*i2hid + j])**2        
        
        #for each input, calculate effect on hidden node by summing all om inputs        
        for j in range(i2hid):
            one_learned.append(in_to_hidden.params[i*i2hid + j]/math.sqrt(sumOfWeights))
            
        learned.append(one_learned)
        
    

    
    fig3, axes = plt.subplots(nrows=int(math.sqrt(len(learned))), ncols=(int(math.sqrt(len(learned)))))
    for dat, ax in zip(learned, axes.flat):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(np.reshape(dat,(myCleaver.sizePatches,myCleaver.sizePatches)), cmap=plt.cm.gray, interpolation='nearest')

    #print(len(myCleaver.patchDataBase))    
    #print(myCleaver.getImages)
    #getTrainingSet()

#    print 'np.array(net.activate(myCleaver.concImgArray.T[0]/256.0))'
#    print net.activate(myCleaver.concImgArray.T[0]/256.0)
#    print 'np.array(net.activate(myCleaver.concImgArray.T[1]/256.0))'
#    print net.activate(myCleaver.concImgArray.T[1]/256.0)
#    print('')  
#    
#    print('the two inputs')
#    print(ds.getSample(0))
#    print(ds.getSample(1))   
#    print('')
#    
#    print('original inputs')
#    print(myCleaver.concImgArray.T[0]/256.0)
#    print(myCleaver.concImgArray.T[1]/256.0)
#    print('')
#    
#    print('first activation')    
#    print(net.activate(myCleaver.concImgArray.T[0]/256.0))
#    print('second activation')  
#    print(net.activate(myCleaver.concImgArray.T[1]/256.0))

    




#    print(net.params)
#    print(net)
#    print(net.outmodules)    
#    
#    
#    net = buildNetwork(4,2,4, bias = True)
#    print('simple net activation')    
#    print(net.activate((1,2,3,4)))
#    print('simple net activation2')    
#    print(net.activate((5,4,3,2)))    
#    print('')
#    
#    for mod in net.modules:
#      print "Module:", mod.name
#      if mod.paramdim > 0:
#        print "--parameters:", mod.params
#      for conn in net.connections[mod]:
#        print "-connection to", conn.outmod.name
#        if conn.paramdim > 0:
#           print "- parameters", conn.params
#      if hasattr(net, "recurrentConns"):
#        print "Recurrent connections"
#        for conn in net.recurrentConns:             
#           print "-", conn.inmod.name, " to", conn.outmod.name
#           if conn.paramdim > 0:
#              print "- parameters", conn.params    
    
if __name__ == '__main__':
    main()