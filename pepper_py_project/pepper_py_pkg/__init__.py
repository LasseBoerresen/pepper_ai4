import math
import numpy as np
import scipy as sp
from scipy import ndimage, misc
import skimage
from skimage import data, filter, io
import matplotlib.pyplot as plt

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
            self.Z.append( (self.A[i].dot( self.W[i] ))+B[i] )#1x64 dot 64x32 -> 1x32 
            #initialize activation list
            a = []           
            #calculate sigmoid for every entry in the latest entry in Z
            for j in range(self.nn[i+1]): 
                a.append(1.0/(1.0+math.exp(-self.Z[i][j])))
            #save  the activations
            self.A.append(np.array(a))
            

    def backPropagation(self):
        self.E = []
        #calculate error for output layer first        
        e = []   
        #for all elements of output layer, calculate error derivative, e = delta.
        for j in range(self.nn[-1]):
            #error for each output node given by: e^(n_l) = <(y-a^(l)),f'(z^(n_l)>, with f'() given by the derivative of the sigmoid function: f'(z) a^(l)*(1-a^(l))
            e.append(-(self.A[0][j]-self.A[-1])*(self.A[-1][j]*(1-self.A[-1][j])) )
        
        self.E.append(e)
        
        
        for i in range(self.numHidden):
            e = []            
                
            self.E.append(np.array(e))










class imageCleaver:
    def __init__(self):
        self.numImages = 1
        self.numPatches = 3
        self.sizePatches = 8
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
        
        
    def concatenateImages(self):        
#        imgArray = np.array([[]])
        concImgArrayT = np.transpose(self.concImgArray)
        
        for i in range(self.numPatches):     
            
            concImgArrayT[i] = self.patchDataBase[i].ravel()
        self.concImgArray = np.transpose(concImgArrayT)
        print(self.concImgArray.shape)
#        print(concImgArrayT)
#            for j in range(self.sizePatches):
#                for k in range(self.sizePatches):
#                                        
#                    self.concImgArray[i] = 

    def getImages(self):    
        return 3

def main():
    myCleaver = imageCleaver()
    #print(myCleaver.getImages)
    #getTrainingSet()

if __name__ == '__main__':
    main()