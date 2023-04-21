import os
seed = 50
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
from hopls import matricize, qsquared, HOPLS
import random
from hankelization import *
from itertools import product
from tensorly.tenalg import mode_dot
random_seed = 5
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
#device='cuda:0'
device='cpu'

def plot_y3D(Y):
    for i,data in [('Y',Y)]:
        with PdfPages(i+'.pdf') as pdf:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(data[50:160,0],
                    data[50:160,1],
                    data[50:160,2],
                    label='parametric curve')
            ax.grid(False)
            ax.set_title(i)
            ax.xaxis.set_rotate_label(False)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            ax.view_init(elev=20, azim=45)
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            plt.show()
            
            
def plot_x(X):
    f, ax = plt.subplots(figsize=(15, 5))
    ax = sns.heatmap(X[50:160,1,:].T)
    ax.set_xlabel('Time')
    ax.set_ylabel('Channels')
    ax.set_title('X')
    plt.show()
    

def plot_y(Y, till=7500):
    fig,ax = plt.subplots(1,1,figsize=(15,4))
    if till<1000:
        maximum = Y.std() 
    else:
        maximum = Y.mean() 
    for i in range(Y.shape[1]):
        ax.plot(Y[100:till,i] - Y[100:till,i].mean() + maximum*i)
    plt.yticks(ticks=[maximum*i  for i in range(Y.shape[1])], labels=[f'ch{i}' for i in range(Y.shape[1])])
    plt.show()
    
        
def metric_nmse_(Y_pred, Y):
    """
    nRMSE 
    """
    Y_base = Y.mean(axis=0)
    return np.mean(np.sqrt(np.mean((Y - Y_pred) ** 2, axis=0)) / np.sqrt(np.mean((Y - Y_base) ** 2, axis=0)))


def metric_nmse_hopls(Y, Y_pred):
    """
    1-nRMSE for learning hopls model
    """
    return 1 - metric_nmse_(Y_pred, Y)


def compute_rmse_pls(X, Y, Xtest_ar, Ytest_ar, k, hankelization = None):
    """
    Input:
        tdata, tlabel - X and Y train
        vdata, vlabel - X and Y test
        k - number of components in PLS
        hankelization - None is data without hankelization, list [hankelization_y_train, hankelization_y_test], where hankelization_y_... is a class hankelization for returning initial form of target data
    Output:
        rmse_test, rmse_train - nRMSE for PLS on test and train
    """
    pls = PLSRegression(n_components=k)
    X = X.reshape(X.shape[0],-1)
    Xtest_ar = Xtest_ar.reshape(Xtest_ar.shape[0],-1)
    pls.fit(X, Y.reshape(Y.shape[0],-1))
    if hankelization != None:
        y_tr_pred, y_te_pred = pls.predict(X).reshape(Y.shape), pls.predict(Xtest_ar).reshape(Ytest_ar.shape)
        y_tr, y_te = Y, Ytest_ar
        y_tr_pred, y_te_pred = hankelization[0].back(y_tr_pred), hankelization[1].back(y_te_pred)
        y_tr, y_te = hankelization[0].back(y_tr), hankelization[1].back(y_te)
        rmse_train = metric_nmse_(y_tr_pred, y_tr)
        rmse_test = metric_nmse_(y_te_pred, y_te)
    else:
        rmse_train = metric_nmse_(pls.predict(X),Y)
        rmse_test = metric_nmse_(pls.predict(Xtest_ar), Ytest_ar)

    return rmse_test, rmse_train


def compute_rmse_hopls(tdata, tlabel, vdata, vlabel, Ln, Km, R_max=30, hankelization = None, train=False):
    """
    Input:
        tdata, tlabel - X and Y train
        vdata, vlabel - X and Y test
        Ln - list, the ranks for the decomposition of X: [L2, ..., LN].
        Km - list, the ranks for the decomposition of Y: [K2, ..., KM].
        R - int, The maximal number of latent vectors.
        hankelization - None is data without hankelization, 
                        list [hankelization_y_train, hankelization_y_test], where hankelization_y_... is a class hankelization for returning initial form of target data
    Output:
        rmse_test, rmse_train - float, nRMSE for PLS on test and train
        R - int, the best number of latent vectors 
    """
    model = HOPLS(R_max, Ln, Km, metric=metric_nmse_hopls, epsilon=1e-3)
    model = model.fit(tdata, tlabel)
    prediction, R, prediction_scores = model.predict(vdata, vlabel)
    if hankelization != None:
        y_pred, y = prediction, vlabel
        y_pred, y = hankelization[1].back(y_pred), hankelization[1].back(y)
        rmse_test = metric_nmse_(y_pred, y)
    else:
        rmse_test = metric_nmse_(prediction, vlabel)
    if train:
        prediction, _, prediction_scores = model.predict(tdata, tlabel)
        if hankelization != None:
            y_pred, y = prediction, tlabel
            y_pred, y = hankelization[0].back(y_pred), hankelization[0].back(y)
            rmse_train = metric_nmse_(y_pred, y)
        else:
            rmse_train = metric_nmse_(prediction, tlabel)
    else:
        rmse_train = None
    return rmse_test, rmse_train, R


def tensorization(X_tr, Y_tr, X_te, Y_te, lenths, axes):
    """
    Input:
        X_tr, Y_tr, X_te, Y_te - ndarray, dataset
        lenths - list of lists, [lenth_x, lenth_y] where lenth_. is a list of parameters for MDT for each axis
        axes - list of lists, [axes_x, axes_y] where axes_. is a list of axes for MDT
    Output:
        data - list, tensorized data [X_tr_ind, Y_tr_ind, X_te_ind, Y_te_ind] 
        hankelizations - list, hankelization classes [H_x_tr, H_x_te, H_y_tr, H_y_te]
    """
    H_x_tr = hankelization(lenths[0],axes[0])
    H_x_te = hankelization(lenths[0],axes[0])
    if len(lenths[1])==0:
        H_y_tr = None
        H_y_te = None
        Y_tr_ind = Y_tr
        Y_te_ind = Y_te        
    else:
        H_y_tr = hankelization(lenths[1],axes[1])
        H_y_te = hankelization(lenths[1],axes[1])
        Y_tr_ind = H_y_tr.forward(Y_tr)
        Y_te_ind = H_y_te.forward(Y_te) 

    X_tr_ind = H_x_tr.forward(X_tr)
    X_te_ind = H_x_te.forward(X_te)
    
    return [X_tr_ind, Y_tr_ind, X_te_ind, Y_te_ind], [H_x_tr, H_x_te, H_y_tr, H_y_te]


def unfold(tensor, mode):
    D = len(tensor.shape)
    perm = [mode]
    for i in range(D):
        if i!=mode:
            perm.append(i)

    return perm, np.transpose(tensor,perm).reshape(tensor.shape[mode], -1)


def svd(tensor, var = 0.99, time_points=10, time_len=10):
    '''
    Output:
    K_min, K_max - lower and higher bounds for dimensions size
    '''
    D = len(tensor.shape)
    indeces = np.random.choice(np.arange(tensor.shape[0]-10), time_points)
    K_min, K_max = [], []
    for k,ind in enumerate(indeces):
        tensor_t = tensor[ind:ind+time_len]
        K_min.append([])
        K_max.append([])
        for mode in range(1,D):
            _, unfolded = unfold(tensor_t, mode)
            u, s, vh = np.linalg.svd(unfolded)
            s = np.array(s)/sum(s)
            l = [sum([1/j for j in range(i,len(s)+1)])/len(s) for i in range(1,len(s)+1)]
            K_min[k].append(((np.array(s)-np.array(l))>0).sum()+1)
            K_max[k].append([sum(s[:i])>var for i in range(1,len(s)+1)].index(True)+1)
    return np.round(np.array(K_min).mean(axis=0)), np.round(np.array(K_max).mean(axis=0))


def tensor_regression(X_tr, Y_tr, X_te, Y_te, ranks_x, hankelization = None):
    """
    Input:
        X_tr, Y_tr, X_te, Y_te - ndarrays, dataset
        ranks_x -list, ranks for tucker regression
        hankelization - None is data without hankelization, 
                        list [hankelization_y_train, hankelization_y_test], where hankelization_y_... is a class hankelization for returning initial form of target data
    Output:
        rmse - float, nRMSE on test
    """
    est = TuckerRegressor(weight_ranks=ranks_x, verbose=0);
    y_pred = np.zeros(Y_te.reshape(Y_te.shape[0],-1).shape)
    y_train = Y_tr.reshape(Y_tr.shape[0],-1)
    for i in range(y_pred.shape[1]):
        est.fit(X_tr, y_train[:, i])
        y_pred[:, i] = est.predict(X_te)
    if hankelization!=None:
        rmse = metric_nmse_(hankelization.back(y_pred.reshape(Y_te.shape)), hankelization.back(Y_te))
    else:
        rmse = metric_nmse_(y_pred.reshape(Y_te.shape), Y_te)
    
    return rmse


def rmse_mpca(X_tr, Y_tr, X_te, Y_te, U_x, U_y=None, hankelization = None):
    ranks = []
    for i in range(U_x.shape[0]):
        if U_x[i,0].shape[0] < 7:
            ranks.append(range(1,U_x[i,0].shape[0]+1))
        elif U_x[i,0].shape[0] < 11:
            ranks.append(range(1,U_x[i,0].shape[0]+1,2))
        elif U_x[i,0].shape[0] < 19:
            ranks.append(range(1,U_x[i,0].shape[0]+1,3))
        else:
            ranks.append(range(1,U_x[i,0].shape[0]+1,4))
    
    for i in range(U_x.shape[0]):
        X_tr = mode_dot(X_tr,U_x[i,0],i+1)
        X_te = mode_dot(X_te,U_x[i,0],i+1)
        
    if U_y!=None:
        for i in range(U_y.shape[0]):
            Y_tr = mode_dot(Y_tr,U_y[i,0],i+1)
            
    y_train = Y_tr.reshape(Y_tr.shape[0],-1)
    
    MSE_test = {}
    for rank in tqdm(product(*ranks)):
        est = TuckerRegressor(weight_ranks=rank, verbose=0);
        
        y_pred = np.zeros((Y_te.shape[0], y_train.shape[1]))
        for i in range(y_pred.shape[1]):
            est.fit(X_tr, y_train[:, i])
            y_pred[:, i] = est.predict(X_te)
        
        y_pred = y_pred.reshape(Y_te.shape[0], *Y_tr.shape[1:])
        if U_y!=None:
            for i in range(U_y.shape[0]):
                y_pred = mode_dot(y_pred,U_y[i,0].T,i+1)
            
        if hankelization!=None:
            rmse = metric_nmse_(hankelization.back(y_pred), hankelization.back(Y_te))
        else:
            rmse = metric_nmse_(y_pred, Y_te)
        MSE_test[f'{rank}'] = rmse
    
    return MSE_test
    
        
# def get_bootstrap(X, Y):
#     n = X.shape[0]
#     idxs = np.random.choice(np.arange(n), size=n)
#     return X[idxs], Y[idxs]


# def CCA_PLS_test(Ytrain_ar, Ytest_ar, Xtrain_ar, Xtest_ar, k_max, algorithm = 'CCA', hankelization=None, plot=True):
#     try:
#         Ytrain_ar, Ytest_ar = Ytrain_ar.cpu().detach().numpy(), Ytest_ar.cpu().detach().numpy()
#         Xtrain_ar, Xtest_ar = Xtrain_ar.cpu().detach().numpy(), Xtest_ar.cpu().detach().numpy()
#     except:
#         Ytrain_ar, Ytest_ar, Xtrain_ar, Xtest_ar = Ytrain_ar, Ytest_ar, Xtrain_ar, Xtest_ar
#     n_restarts = 10
#     rmse = []
#     for k in tqdm(range(1, k_max)):#tqdm(range(1, k_max)):
#         if algorithm == 'CCA':
#             pls = CCA(n_components=k)
#         else:
#             pls = PLSRegression(n_components=k)
#         tr = []
#         te = []
#         for i in range(n_restarts):
#             X, Y = get_bootstrap(Xtrain_ar, Ytrain_ar)
#             pls.fit(X, Y.reshape(Y.shape[0],-1))
#             if indexing:
#                 y_tr_pred, y_te_pred = pls.predict(X).reshape(Y.shape), pls.predict(Xtest_ar).reshape(Ytest_ar.shape)
#                 y_tr, y_te = Y, Ytest_ar
#                 y_tr_pred, y_te_pred = hankelization[0].back(y_tr_pred), hankelization[1].back(y_te_pred)
#                 y_tr, y_te = hankelization[0].back(y_tr), hankelization[1].back(y_te)
#                 tr.append(metric_nmse_(y_tr_pred, y_tr))
#                 te.append(metric_nmse_(y_te_pred, y_te))
#             else:
#                 tr.append(metric_nmse_(pls.predict(X),Y))
#                 te.append(metric_nmse_(pls.predict(Xtest_ar), Ytest_ar))
#         rmse.append((k, tr, te))

#     x = [r[0] for r in rmse]
#     y1_mean = np.array([np.mean(r[1]) for r in rmse])
#     y2_mean = np.array([np.mean(r[2]) for r in rmse])
#     y1_std = np.array([np.std(r[1]) for r in rmse])
#     y2_std = np.array([np.std(r[2]) for r in rmse])
#     if plot:
#         plt.figure(figsize=(6, 5))

#         plt.plot(x, y1_mean, label='train')
#         plt.fill_between(x, y1_mean - y1_std, y1_mean + y1_std, alpha=0.3)

#         plt.plot(x, y2_mean, label='test')
#         plt.fill_between(x, y2_mean - y2_std, y2_mean + y2_std, alpha=0.3)
#         plt.ylabel(r'sRMSE')
#         plt.xlabel(algorithm + r'latent space dimensionality, $l$')

#         plt.legend(loc='upper right')
#         plt.tight_layout()
#         plt.show()

#         print(np.argmin(np.array(y2_mean))+1,'components with nRMSE =', min(y2_mean))
#     return min(y2_mean)
