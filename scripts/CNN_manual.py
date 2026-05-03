# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 13:41:32 2026

@author: jeco
"""

#import numpy as np
import cupy as cp

#from scipy import signal
from cupyx.scipy import signal

from keras import utils
#from keras.datasets import mnist

import h5py

from pathlib import Path
#Based on and help from: www.youtube.com/watch?v=Lakz2MoHy6o
#Takes a very long time to process, switched to pytorch method

#%% Paths
script_dir = Path(__file__).resolve().parent

dataLoc=script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_RowFFT\TrainRow.hdf5'
#dataLoc= script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_ColFFT\TrainCol.hdf5'

dataLoc=str(dataLoc)

#%% Class Templates
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return cp.multiply(output_gradient, self.activation_prime(self.input))


#%% Class Layers
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width= input_shape
        self.depth= depth
        self.input_shape= input_shape
        self.input_depth= input_depth
        self.output_shape= (depth, input_height- kernel_size+1, input_width-kernel_size+1)
        self.kernels_shape= (depth, input_depth, kernel_size, kernel_size)
        
        #self.kernels= cp.random.rand(*self.kernels_shape)
        #self.biases= cp.random.rand(*self.output_shape)
        
        #try to fix Nan output
        self.kernels= cp.random.randn(*self.kernels_shape) * cp.sqrt(2.0 /(self.input_depth * kernel_size**2))
        self.biases= cp.zeros(self.output_shape)
        
    def forward(self, input):
        self.input= input
        self.output= cp.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient= cp.zeros(self.kernels_shape)
        input_gradient= cp.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j]= signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], 'full')
        
        self.kernels-= learning_rate * kernels_gradient
        self.biases-= learning_rate * output_gradient
        return input_gradient


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Original Random Normal
        #self.weights = cp.random.randn(output_size, input_size)
        #self.bias = cp.random.randn(output_size, 1)
        
        # He Normal Initialization 
        self.weights= cp.random.randn(output_size, input_size) * cp.sqrt(2.0 / input_size)
        self.bias= cp.zeros((output_size, 1))
        

    def forward(self, input):
        self.input = input
        return cp.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        #print(self.input.T.shape)
        #print(output_gradient.shape)
        weights_gradient = cp.dot(output_gradient, self.input.T)
        input_gradient = cp.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape= input_shape
        self.output_shape= output_shape
        
    def forward(self, input):
        return cp.reshape(input, self.output_shape)    
    
    def backward(self, output_gradient, learning_rate):
        return cp.reshape(output_gradient, self.input_shape)
    
#%% Loss

def binary_cross_entropy(y_true, y_pred):
    y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
    return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / cp.size(y_true)

def catCross_entropy(y_true, y_pred):
    y_pred= cp.clip(y_pred, 1e-15, 1 - 1e-15)
    return -cp.sum(y_true * cp.log(y_pred))

def catCross_entropy_prime(y_true, y_pred):
    y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
    return -y_true / y_pred

#%% Activation

class Sigmoid(Activation):
    def __init__(self):
        
        def sigmoid(x):
            return 1/(1+cp.exp(-x))
        
        def sigmoid_prime(x):
            s= sigmoid(x)
            return s * (1-s)
        
        super().__init__(sigmoid, sigmoid_prime)
        
class ReLU(Activation):
    def __init__(self):
        
        def relu(x):
            return cp.maximum(0, x)

        def relu_prime(x):
            return cp.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def forward(self, input):
        exps = cp.exp(input - cp.max(input))
        self.output = exps / cp.sum(exps)
        return self.output

    def backward(self, output_gradient, learning_rate):
        #n = cp.size(self.output)
        
        out_flat= self.output.reshape(-1, 1)
        deriv = cp.diagflat(out_flat) - cp.dot(out_flat, out_flat.T)
        out= cp.dot(deriv, output_gradient)
        #print(out.shape)
        return  out


#%% Preprocess
def preprocess_data2(x,y):
    all_indices=cp.arange(len(y))
    all_indices= cp.random.permutation(all_indices)
    
    x,y = x[all_indices], y[all_indices]
    x= x.reshape(len(x),3,343,434)
    
    y= cp.array(utils.to_categorical(y.tolist()))
    y= cp.array(y.reshape(len(y),11,1))
    return x, y

#%% Training
def train(x_train, y_train, network):
    epochs= 100
    learning_rate= 0.5# increased because slow
    print('Starting Training...')
    for e in range(epochs):
        error= 0
        for x,y in zip(x_train, y_train):
            output= x
            for layer in network:
                output= layer.forward(output)
                
                #try to fix Nan output
                # if cp.isnan(output).any():
                #     print(f"NaN detected in forward pass of {layer.__class__.__name__}")
            
            error += catCross_entropy(y, output)
            print(f'Running error: {error}') # used primarily as a sanity check that it hasn't crashed
            #grad= catCross_entropy_prime(y, output)
            grad= output - y#try to fix Nan output
            for layer in reversed(network):
                grad= layer.backward(grad, learning_rate)
                
        error /= len(x_train)
        print(f"{e+1}/{epochs}, error={error}")
        
    return network


#%% Testing Inputs 


classNames=["walking_towards", "walking_away", "picking_obj", "Bending", 
            "sitting", "kneeling", "crawling", "walking_on_toe", "limping_RL",
            "short_steps", "scissor_gait"]
rawDataTrain= h5py.File(dataLoc)
xdata=cp.array(rawDataTrain['xdata'][:])
ydata=cp.array(rawDataTrain['ydata'][:])
uClass, invIndx = cp.unique(ydata, return_inverse=True)

#y_train2= cp.eye(len(uClass))[invIndx]
#x_train2= xdata.reshape(545,343*434*3, order='F')
y_train2= invIndx
x_train2= xdata

rawDataTrain.close()

x_train2, y_train2= preprocess_data2(x_train2, y_train2)

#%% Attempt 1
network=[
    Convolutional((3,343,434),3,5),
    ReLU(),
    Reshape((5,341,432),(5*341*432,1)),
    Dense(5*341*432,100),
    ReLU(),
    Dense(100,11),
    Softmax()
    ]

network= train(x_train2, y_train2, network)