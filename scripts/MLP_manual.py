# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:14:45 2026

@author: Jeco
"""

import h5py
import matplotlib.pyplot as plt

import cupy as cp #hopefully this makes it faster than numpy

from pathlib import Path
# based on and help from: https://ee541.usc-ece.com/guides/hw/hw07/index.html

#%% Paths
script_dir = Path(__file__).resolve().parent

#dataLoc=script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_RowFFT\TrainRow.hdf5'
#dataLoc=script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_RowFFT\TestRow.hdf5'

dataLocTrn= script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_ColFFT\TrainCol.hdf5'
dataLocTst= script_dir.parent / r'sorted\zzz_outputs_m120dBFloor_sitPulseRef_Centered_ColFFT\TestCol.hdf5'

dataLocTrn=str(dataLocTrn)
dataLocTst=str(dataLocTst)

#%% Inputs 
# Loading times
# - 5.22sec with numpy
# - 3.5sec with cupy

classNames=["walking_towards", "walking_away", "picking_obj", "Bending", 
            "sitting", "kneeling", "crawling", "walking_on_toe", "limping_RL",
            "short_steps", "scissor_gait"]

rawDataTrain= h5py.File(dataLocTrn)
rawDataTest= h5py.File(dataLocTst)

xdata=cp.array(rawDataTrain['xdata'][:])
ydata=cp.array(rawDataTrain['ydata'][:])
uClass, invIndx = cp.unique(ydata, return_inverse=True)
one_hot= cp.eye(len(uClass))[invIndx]
dataTrain={"xdata": xdata.reshape(545,343*434*3, order='F'),
           "ydata": one_hot
           }

del xdata, invIndx, ydata, one_hot

xdata=cp.array(rawDataTest['xdata'][:])
ydata=cp.array(rawDataTest['ydata'][:])
uClass, invIndx = cp.unique(ydata, return_inverse=True)
one_hot= cp.eye(len(uClass))[invIndx]
dataTest={"xdata": xdata.reshape(131,343*434*3, order='F'),
           "ydata": one_hot
           }
del xdata, invIndx, ydata, one_hot

print(f'Original Class Numbers: {uClass}')
rawDataTest.close()
rawDataTrain.close()


# #%% Data Conversions
# image= dataTrain['xdata'][0]
# r= image[:,:,0];
# g= image[:,:,1];
# b= image[:,:,2];
# imgRav=np.ravel(image,'F')# convert with option 'F'
# #indexing is like this: imgRav[(343+433*343)+(343+433*343)+(r_indx+c_indx*343)] 
# # shows that we are searching through blue layer at that row and col index
# img2=cp.asnumpy(cp.dstack((r,g,b)))


#%% Layer activations
def relu(s):
    a=cp.maximum(s,0)
    return a
    
def relu_deriv(s):
    grad_s= (s>0).astype(int)#not including product with grad_a       
    return grad_s

def tanh(s):
    a= cp.tanh(s)
    return a

def tanh_deriv(s):
    grad_s= (1-(tanh(s))**2)# remember this is element wise multiplication and squaring
    return grad_s

def softmax(s):
    exps = cp.exp(s - cp.max(s, axis=1, keepdims=True))
    return exps / cp.sum(exps, axis=1, keepdims=True)
    

#%% Cost function: cross entropy
def cross_entropy(y_true, y_pred):# true is y and pred is a in 1.3.1 equation
    """Numerically stable implementation of cross-entropy loss"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = cp.clip(y_pred, epsilon, 1 - epsilon)
    #return -cp.sum(y_true * cp.log(y_pred), axis=0)/y_true.shape[0]# included scaling for avg
    return -cp.mean(cp.sum(y_true * cp.log(y_pred), axis=1))


#%% Optimizer: Learning Rate Scheduling classes

# Constant
class constLr:
    def __init__(self, initial_lr):
        self.lr=initial_lr
    def calcLr(self, epochs):
        return self.lr
    
# Step decay (every 10 epochs, reduce by half)
class stepLr:
    def __init__(self, initial_lr):
        self.lr=initial_lr
    def calcLr(self, epochs):
        return self.lr * (0.5 ** (cp.floor(epochs / 10)))
    
# Exponential decay
class expLr:
    def __init__(self, initial_lr):
        self.lr=initial_lr
    def calcLr(self, epochs):
        return self.lr * cp.exp(-0.03 * epochs)
    
# 1/t decay
class TthLr:
    def __init__(self, initial_lr):
        self.lr=initial_lr
    def calcLr(self, epochs):
        return self.lr / (1 + 0.05 * epochs)

#%% Network Class
class NeuralNetwork:
    def __init__(self, layers=[446586, 256, 128, 11], activation='relu'): # (343*434*3) class inputs,  11 class outputs
    #chose hidden layers to bottle neck thus preventing greater number of parameters
    # - risk of losing characteristics and interactions like neihboring pixel correlations
    # - experimented with higher number of neurons [512,256] instead of [256,128], took longer to process with no diff
        self.weights = []
        self.biases = []
        self.act_func = relu if activation == 'relu' else tanh
        self.act_deriv = relu_deriv if activation == 'relu' else tanh_deriv
        
        for i in range(len(layers) - 1):
            # f) Initialization used Gaussian Distribution with scaling of 0.01
            limit = cp.sqrt(2 / layers[i])#apply He-Xavier to prevent exploding/vanishing gradients
            w = cp.random.randn(layers[i], layers[i+1]) * limit
            b = cp.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.a = [X] # Activations
        self.z = []  # Pre-activations
        
        for i in range(len(self.weights) - 1):
            z = cp.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            self.a.append(self.act_func(z))
            
        # Output or last layer uses softmax
        z_out = cp.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z_out)
        self.a.append(softmax(z_out))
        return self.a[-1]

    def backward(self, X, y_true, lr):
        m = X.shape[0]
        # Cross-Entropy + Softmax gradient: (y_pred - y_true)
        dz = self.a[-1] - y_true 
        
        for wIndx in reversed(range(len(self.weights))):
            dw = cp.dot(self.a[wIndx].T, dz) / m
            db = cp.sum(dz, axis=0, keepdims=True) / m
            
            # Prepare dz for the next (previous) layer
            if wIndx > 0:
                dz = cp.dot(dz, self.weights[wIndx].T) * self.act_deriv(self.z[wIndx-1])
            
            # Gradient Update
            self.weights[wIndx] -= lr * dw
            self.biases[wIndx] -= lr * db
            
#%% Training and Plotting Functions

def train(X, y, lr, activation, epochs=100, batch_size=16):
    #anything above 50 epochs is overfitting due the "small" amount of images to train off of
    #- kept at 30 epochs
    #original was 172, trying smallest of 1 with expectation that training time will increase
    #- tried batch_size=1 with lr=0.5 did not work well stopped early
    #- seems like batch sizes of 2^n are preferable with cupy, maybe 64 is best
    
    model = NeuralNetwork(activation=activation)
    n_samples = X.shape[0]
    
    # Split data to Training(516) and Validation(29)
    indicesVal = cp.random.permutation(n_samples)
    XVal=X[indicesVal[0:516]]
    yVal=y[indicesVal[0:516]]
    #intVal=cp.argmax(yVal)
    intVal= cp.argmax(yVal, axis=1)
    
    X=X[indicesVal[29::]]
    y=y[indicesVal[29::]]
    #intTrn=cp.argmax(y)
    intTrn= cp.argmax(y, axis=1)
    n_samples = X.shape[0] #update to new size
    
    # Decay Learning Rate
    lrDecay=expLr(lr)# will be applied to current epoch during backward call
    
    accListVal=[]
    accListTrn=[]
    
    for epoch in range(epochs):
        # Shuffle data
        indices = cp.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward(calc next inputs)
            model.forward(X_batch)
            
            # Backward(calc deriv) 
            model.backward(X_batch, y_batch, lrDecay.calcLr(epoch))
            
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            y_pred_full = model.forward(X)
            loss = cross_entropy(y, y_pred_full)
            print(f"Epoch {epoch+1}, Loss: {loss}, LR: {lrDecay.calcLr(epoch)}")
        
        # Forward with Training 
        aTrn = model.forward(X)
        predTrn= cp.argmax(aTrn, axis=1)
        
        # Forward with Validation
        aVal=model.forward(XVal)
        predVal=cp.argmax(aVal, axis=1)
        
        # Compute Accuracy 
        accListVal.append(cp.sum(predVal == intVal)/XVal.shape[0])
        accListTrn.append(cp.sum(predTrn == intTrn)/X.shape[0])
        
        
    return model, cp.asnumpy(cp.array(accListVal)), cp.asnumpy(cp.array(accListTrn))

def plot_accuracy_curves(train_acc, val_acc):
    # tempTrn= cp.asnumpy(cp.array(train_acc))
    # tempVal= cp.asnumpy(cp.array(val_acc))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plots')
    plt.grid(True)
    plt.legend()
    plt.show()

    

#%% config1) activation=relu and lr=0.5
out1= train(X=dataTrain['xdata'], y=dataTrain['ydata'], activation='relu' ,lr=0.5)
plot_accuracy_curves(out1[2], out1[1])

#%% config2) activation=relu and lr=0.1
out2= train(X=dataTrain['xdata'], y=dataTrain['ydata'], activation='relu' ,lr=0.1)
plot_accuracy_curves(out2[2], out2[1])

#%% config3) activation=relu and lr=0.02
out3= train(X=dataTrain['xdata'], y=dataTrain['ydata'], activation='relu' ,lr=0.02)
plot_accuracy_curves(out3[2], out3[1])

print('testing expLr')
raise SystemExit("Stopping execution here.")
#%% Check Test
XTst=dataTest['xdata']
yTst=dataTest['ydata']

# Forward with Training 
aTst = out3[0].forward(XTst)
predTst= cp.argmax(aTst, axis=1)
intTst=cp.argmax(yTst, axis=1)

# Compute Accuracy 
accFin=cp.sum(predTst == intTst)/XTst.shape[0]
print(f'Accurracy: {accFin}')