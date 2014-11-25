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
        self.numPatches = 1000
        self.sizePatches = 8#8
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



def main():
    #sample patches from image database
    myCleaver = imageCleaver()
    
    #instantiate NN with size as defined in image cleaver class. 
    #TODO: size should be input to imageCleaver class.
    net = buildNetwork(myCleaver.sizePatches*myCleaver.sizePatches, myCleaver.sizePatches*myCleaver.sizePatches/4, myCleaver.sizePatches*myCleaver.sizePatches, bias = True)

#    print(net.activate([2, 1]))
    #Put imageCleaver dataset into pyBrain dataset format.
    ds = SupervisedDataSet(myCleaver.sizePatches*myCleaver.sizePatches, myCleaver.sizePatches*myCleaver.sizePatches)
    for i in range(myCleaver.concImgArray.shape[1]):
        ds.addSample(myCleaver.concImgArray.T[i],myCleaver.concImgArray.T[i])
    
#    for inpt, target in ds:
#        print inpt, target
    
    trainer = BackpropTrainer(net, ds)    
    print("activation: ")
    print()
#    print(net.activate([2, 1]))    
#    print(trainer.train())
#    print(trainer.train())
#    print(trainer.train())
#    print(trainer.train())
#    print(trainer.train())
    for i in range(3):    
        print(trainer.train())
    #print(len(myCleaver.patchDataBase))    
    #print(myCleaver.getImages)
    #getTrainingSet()
    #fig, axes = plt.subplots(2, 1, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []}) 
    #fig.subplots_adjust(hspace=0.3, wspace=0.05)

    #plt.imshow(myCleaver.patchDataBase[45],cmap=plt.cm.gray)

    imitationActivations = np.array(net.activate(myCleaver.concImgArray.T[45]))
    imitation = np.reshape(imitationActivations,(8,8))
    plt.imshow(imitation,cmap=plt.cm.gray)
    #plt.show()
    
    
if __name__ == '__main__':
    main()