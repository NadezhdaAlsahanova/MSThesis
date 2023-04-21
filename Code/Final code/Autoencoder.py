import os
seed = 42
os.environ['PYTHONHASHSEED']=str(seed)
import scipy.io
import seaborn as sns
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.cross_decomposition import PLSRegression 
from sklearn.metrics import mean_squared_error as MSE
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import warnings
warnings.simplefilter("ignore")
from tensorly.decomposition import robust_pca
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
# from tensorly.regression.cp_regression import CPRegressor
from tensorly.regression.tucker_regression import TuckerRegressor
import tensorly as tl
from tensorly import random
from hopls import matricize, qsquared, HOPLS
import random
from scipy.linalg import hankel
from qpfs import *
random_seed = 5
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
#!CUBLAS_WORKSPACE_CONFIG=:16:8#:4096:8
device='cuda:0'


def visualize(fig, train_loss, test_loss, clean=True):
    if clean:
        fig.clear()
    plt.plot(train_loss, color='red',label='train')
    plt.plot(test_loss, color='blue',label='test')
    plt.grid(linestyle='--')
    plt.yscale('log')
    plt.legend()
    if clean:
        clear_output(wait=True)
        display(fig);
    else:
        plt.show()
        

class ConvMiniBlock(nn.Module):
    def __init__(self, N_in, N_out, kernel_size=1):
        super().__init__()
        # Encoder specification
        self.cnn = nn.Sequential(
            nn.Conv1d(N_in, N_out, kernel_size),
            nn.BatchNorm1d(N_out),
            nn.GELU(),
            nn.Dropout(p=0.05)) 
        
    def forward(self, x):
        return self.cnn(x) 

class ConvBlock(nn.Module):
    def __init__(self, Ns, kernel_size=1):
        super().__init__()
        # Encoder specification
        modules = []
        for N_in, N_out in zip(Ns[:-1],Ns[1:]):
            modules.append(ConvMiniBlock(N_in,N_out,kernel_size))
        self.block_cnn = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.block_cnn(x) 

    
class AutoEncoder_tensor(nn.Module):
    
    def __init__(self, D, shapes, lstm_shape, y_shape, indexing=False):
        super().__init__()
        '''
        D - int
        shapes - [[N_1,...,l_1],...,[N_(D-2), ..., l_(D-1)]]
        lstm_shape - [N_in,N_out]
        y_shape - int
        '''
        self.D = D
        self.shapes = shapes
        self.lstm_shape = lstm_shape
        self.indexing = indexing
        
        final_shape = 1
        self.enc_cnn = nn.ModuleList()
        for s in shapes:
            self.enc_cnn.append(ConvBlock(s))
            final_shape*=s[-1]

        self.enc_lstm = nn.LSTM(lstm_shape[0], lstm_shape[1], 1)
        
        self.dec_lstm = nn.LSTM(lstm_shape[1], lstm_shape[0], 1)
        self.dec_cnn = nn.ModuleList()
        for s in shapes[::-1]:
            self.dec_cnn.append(ConvBlock(s[::-1]))
        
        self.fc = nn.Linear(lstm_shape[1]*final_shape, y_shape)
        
    def forward(self, images):
        code = self.encode(images)
        y_pred = self.fc(code.reshape(code.shape[0],-1))
        out = self.decode(code)
        return out, y_pred, code
    
    def encode(self, x):
        self.x_shape = x.shape
        if self.indexing:
            shape_x = x.shape
            dims = [0]
            for i in range(2,self.D):
                dims.append(i)
            dims.append(1)
            
            code = torch.permute(x, dims).reshape(shape_x[0], shape_x[2], -1)
            code = self.enc_cnn[0](code)
            for i,cnn in enumerate(self.enc_cnn[1:]):
                code = code.reshape(code.shape[0],code.shape[1], shape_x[i+3], -1)
                code = torch.permute(code, (0,2,3,1))
                code = code.reshape(code.shape[0], code.shape[1], -1)
                code = cnn(code)
            code = code.reshape(code.shape[0],code.shape[1], shape_x[1], -1)
            code = torch.permute(code, (0,2,3,1))
            code = code.reshape(code.shape[0], code.shape[1], -1)
            code = torch.permute(code, (0,2,1))
            code,_ = self.enc_lstm(code)
            code = torch.permute(code, (0,2,1))
            code = code.reshape([shape_x[0], self.lstm_shape[1]]+[s[-1] for s in self.shapes])                
        else:
            if len(x.shape) != 3:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            code = self.enc_cnn[0](x)  
            code,_ = self.enc_lstm(code)
        return code
    
    def decode(self, code):
        if self.indexing:
            shape_code = code.shape
            out = torch.permute(code.reshape(shape_code[0], shape_code[1], -1), (0,2,1))
            out,_ = self.dec_lstm(out)
            out = torch.permute(out,(0,2,1))
            for i,cnn in enumerate(self.dec_cnn):
                out = out.reshape(out.shape[0],-1, shape_code[-1-i])
                out = torch.permute(out, (0,2,1))
                out = cnn(out)
            out = out.reshape(out.shape[0],-1, self.x_shape[1])
            out = torch.permute(out, (0,2,1)).reshape(self.x_shape)               
        else:
            out,_ = self.dec_lstm(code)
            out = self.dec_cnn[0](out)  
            out = out.reshape(self.x_shape) 
        return out
    
    
class AutoEncoder_matrix(nn.Module):
    
    def __init__(self, D, shapes, lstm_shape, y_shape, indexing=False):
        super().__init__()
        '''
        D - int
        shapes - [[N_1,...,l_1],...,[N_(D-2), ..., l_(D-1)]]
        lstm_shape - [N_in,N_out]
        y_shape - int
        '''
        self.D = D
        self.shapes = shapes
        self.lstm_shape = lstm_shape
        self.indexing = indexing
        
        self.enc_cnn = nn.ModuleList()
        for s in shapes:
            self.enc_cnn.append(ConvBlock(s))

        self.enc_lstm = nn.LSTM(lstm_shape[0], lstm_shape[1], 1)
        
        self.dec_lstm = nn.LSTM(lstm_shape[1], lstm_shape[0], 1)
        self.dec_cnn = nn.ModuleList()
        for s in shapes[::-1]:
            self.dec_cnn.append(ConvBlock(s[::-1]))
        
        self.fc = nn.Linear(lstm_shape[1], y_shape)
        
    def forward(self, images):
        code = self.encode(images)
        y_pred = self.fc(code.reshape(code.shape[0],-1))
        out = self.decode(code)
        return out, y_pred, code
    
    def encode(self, x):
        self.x_shape = x.shape
        if self.indexing:
            shape_x = x.shape
            dims = [0]
            for i in range(2,self.D):
                dims.append(i)
            dims.append(1)
            code = torch.permute(x, dims).reshape(shape_x[0], shape_x[2], -1)
            code = self.enc_cnn[0](code)
            for i,cnn in enumerate(self.enc_cnn[1:]):
                code = code.reshape(code.shape[0],code.shape[1], shape_x[i+3], -1)
                code = torch.permute(code, (0,2,3,1))
                code = code.reshape(code.shape[0], code.shape[1], -1)
                code = cnn(code)
            code = code.reshape(code.shape[0], 1, -1)
            code,_ = self.enc_lstm(code)         
            code = code.reshape(x.shape[0], -1)
        else:
            if len(x.shape) != 3:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            code = self.enc_cnn[0](x)  
            code = code.reshape(code.shape[0], 1, -1)
            code,_ = self.enc_lstm(code)
            code = code.reshape(x.shape[0], -1)
        return code
    
    def decode(self, code):
        if self.indexing:
            shape_code = code.shape
            out,_ = self.dec_lstm(code.reshape(code.shape[0], 1, -1))
            out = out.reshape(self.x_shape[0], self.x_shape[1], -1)
            shapes_enc = [s[-1] for s in self.shapes]
            out = torch.permute(out,(0,2,1))
            for i,cnn in enumerate(self.dec_cnn):
                out = out.reshape(out.shape[0],-1, shapes_enc[-1-i])
                out = torch.permute(out, (0,2,1))
                out = cnn(out)
            out = out.reshape(out.shape[0],-1, self.x_shape[1])
            out = torch.permute(out, (0,2,1)).reshape(self.x_shape)               
        else:
            out,_ = self.dec_lstm(code.reshape(code.shape[0], 1, -1))
            out = out.reshape(self.x_shape[0], -1, self.x_shape[2])
            out = self.dec_cnn[0](out)  
            out = out.reshape(self.x_shape) 
        return out
    
    
class AutoEncoder_y_tensor(nn.Module):
    
    def __init__(self, D, shapes, lstm_shape, indexing=False):
        super().__init__()
        '''
        D - int
        shapes - [[N_1,...,l_1],...,[N_(D-2), ..., l_(D-1)]]
        lstm_shape - [N_in,N_out]
        y_shape - int
        '''
        self.D = D
        self.shapes = shapes
        self.lstm_shape = lstm_shape
        self.indexing = indexing
        
        if indexing:
            final_shape = 1
            self.enc_cnn = nn.ModuleList()
            for s in shapes:
                self.enc_cnn.append(ConvBlock(s))
                final_shape*=s[-1]
                
            self.dec_cnn = nn.ModuleList()
            for s in shapes[::-1]:
                self.dec_cnn.append(ConvBlock(s[::-1]))

        self.enc_lstm = nn.LSTM(lstm_shape[0], lstm_shape[1], 1)
        
        self.dec_lstm = nn.LSTM(lstm_shape[1], lstm_shape[0], 1)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, x):
        self.x_shape = x.shape
        if self.indexing:
            if len(x.shape) != 3:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            x = torch.permute(x,(0,2,1))
            code = self.enc_cnn[0](x) 
            code,_ = self.enc_lstm(code)
            code = torch.permute(code, (0,2,1))
        else:
            if len(x.shape) != 2:
                x = x.reshape(x.shape[0], -1)
            x = x.reshape(x.shape[0], 1, -1)
            code,_ = self.enc_lstm(x)
        return code
    
    def decode(self, code):
        if self.indexing:
            shape_code = code.shape
            out = torch.permute(code, (0,2,1))
            out,_ = self.dec_lstm(out)
            out = self.dec_cnn[0](out)
            out = torch.permute(out, (0,2,1)) 
            out = out.reshape(self.x_shape) 
        else:
            out,_ = self.dec_lstm(code)
            out = out.reshape(self.x_shape) 
        return out
    
    
class AutoEncoder_y_matrix(nn.Module):
    
    def __init__(self, D, shapes, lstm_shape, indexing=False):
        super().__init__()
        '''
        D - int
        shapes - [[N_1,...,l_1],...,[N_(D-2), ..., l_(D-1)]]
        lstm_shape - [N_in,N_out]
        y_shape - int
        '''
        self.D = D
        self.shapes = shapes
        self.lstm_shape = lstm_shape
        self.indexing = indexing
        
        if indexing:
            final_shape = 1
            self.enc_cnn = nn.ModuleList()
            for s in shapes:
                self.enc_cnn.append(ConvBlock(s))
                final_shape*=s[-1]
                
            self.dec_cnn = nn.ModuleList()
            for s in shapes[::-1]:
                self.dec_cnn.append(ConvBlock(s[::-1]))

        self.enc_lstm = nn.LSTM(lstm_shape[0], lstm_shape[1], 3)
        
        self.dec_lstm = nn.LSTM(lstm_shape[1], lstm_shape[0], 3)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, x):
        self.x_shape = x.shape
        if self.indexing:
            if len(x.shape) != 3:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            x = torch.permute(x,(0,2,1))
            code = self.enc_cnn[0](x).reshape(self.x_shape[0], 1, -1) 
            code,_ = self.enc_lstm(code)
            code = code.reshape(code.shape[0], -1)
        else:
            if len(x.shape) != 2:
                x = x.reshape(x.shape[0], -1)            
            x = x.reshape(x.shape[0], 1, -1)
            code,_ = self.enc_lstm(x)
            code = code.reshape(x.shape[0], -1)
        return code
    
    def decode(self, code):
        if self.indexing:
            shape_code = code.shape
            out,_ = self.dec_lstm(code.reshape(code.shape[0], 1, -1))
            out = out.reshape(out.shape[0],-1, self.x_shape[1])
            out = self.dec_cnn[0](out)
            out = torch.permute(out, (0,2,1)) 
            out = out.reshape(self.x_shape) 
        else:
            out,_ = self.dec_lstm(code.reshape(code.shape[0], 1, -1))
            out = out.reshape(self.x_shape) 
        return out
    

def loss_y():
    def loss_(y, y_pred):
        return (((y-y_pred)**2).mean(axis=0) / ((y)**2).mean(axis=0)).mean()
    return loss_

def my_loss(l):
    def loss_(X, y, out, y_pred):
        return ((X-out)**2).mean() + l * ((y-y_pred)**2).mean()
    return loss_

def my_loss_Q(l,m):
    def loss_(X, y, out, y_pred, code):
        return ((X-out)**2).mean() + l * ((y-y_pred)**2).mean() + m * torch.mean(compute_pearson(code.reshape(X.size(0),-1,1)))
    return loss_


def train_epoch(model, train_loader, optimizer, loss_fn, scheduler):
    epoch_loss = []
    for x,y in train_loader:
        out, y_pred, code = model(x)
        y_pred = y_pred.reshape(y.shape)
        optimizer.zero_grad()
        loss = loss_fn(x, y, out, y_pred)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        epoch_loss.append(loss.item())
    return np.array(epoch_loss).mean()

    
def test_epoch(model, test_loader, loss_fn):
    with torch.no_grad():
        epoch_loss = []
        for x,y in test_loader:
            out, y_pred, code = model(x)
            y_pred = y_pred.reshape(y.shape)
            loss = loss_fn(x, y, out, y_pred)
            epoch_loss.append(loss.item())
    return np.array(epoch_loss).mean()


def train(model, X_tr, Y_tr, X_te, Y_te, lr, step_size, batch_size, epochs, loss_fn, plot=True):
    train = data_utils.TensorDataset(X_tr, Y_tr)
    test = data_utils.TensorDataset(X_te, Y_te)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    optimizer_cls = optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    if plot:
        fig = plt.figure(figsize=(10,5))
    epoch_train_loss = []
    epoch_test_loss = []

    for epoch in range(epochs):
        epoch_train_loss.append(train_epoch(model, train_loader, optimizer, loss_fn, scheduler))
        epoch_test_loss.append(test_epoch(model, test_loader, loss_fn))
        if plot:
            visualize(fig, epoch_train_loss, epoch_test_loss);
    return model
