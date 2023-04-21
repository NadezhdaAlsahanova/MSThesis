"""
File for finding the best parameters for MPCA+Tensor Regression  model
"""

import os
import scipy.io
import seaborn as sns
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
from utils import *
from hankelization import * 
import multiprocessing 
import time
random_seed = 5
random.seed(random_seed)
np.random.seed(random_seed)
#device='cuda:0'
device='cpu'


def wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n):
    if 'EEG' in dataset_exp:
        U_x = scipy.io.loadmat('MPCA/60_X.mat')['ans']
    else:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_X.mat')['ans']
    
    MSE_mpca = rmse_mpca(X_train_n, Y_train_n, X_test_n, Y_test_n, U_x)
    with open(f"experiments/{dataset_exp}_wt_mpca.npz", "wb") as f:
        np.savez(f, **MSE_mpca)
    
    
def ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n):
    data, H_t = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10],[10]], [[0],[0]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data
    if 'EEG' in dataset_exp:
        U_x = scipy.io.loadmat('MPCA/60_X_ht.mat')['ans']
        U_y = scipy.io.loadmat('MPCA/60_Y_ht.mat')['ans']
    elif '86' in dataset_exp:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_ht_X.mat')['ans']
        U_y = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_ht_Y.mat')['ans']
    else:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_X_ht.mat')['ans']
        U_y = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_Y_ht.mat')['ans']
        
    MSE_mpca = rmse_mpca(X_train_n, Y_train_n, X_test_n, Y_test_n, U_x, U_y, H_t[-1])
    with open(f"experiments/{dataset_exp}_ht_mpca.npz", "wb") as f:
        np.savez(f, **MSE_mpca)

    
def hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n):
    data, H_s = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[2],[]], [[2],[]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data
    if 'EEG' in dataset_exp:
        U_x = scipy.io.loadmat('MPCA/60_X_hs.mat')['ans']
    elif '86' in dataset_exp:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_hs_X.mat')['ans']
    else:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_X_hs.mat')['ans']
        
    MSE_mpca = rmse_mpca(X_train_n, Y_train_n, X_test_n, Y_test_n, U_x)
    with open(f"experiments/{dataset_exp}_hs_mpca.npz", "wb") as f:
        np.savez(f, **MSE_mpca)
    
    
    
def hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n):
    data, H_b = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10,2],[10]], [[0,2],[0]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data
    
    if 'EEG' in dataset_exp:
        U_x = scipy.io.loadmat('MPCA/60_X_hb.mat')['ans']
        U_y = scipy.io.loadmat('MPCA/60_Y_hb.mat')['ans']
    elif '86' in dataset_exp:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_hb_X.mat')['ans']
        U_y = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_hb_Y.mat')['ans']
    else:
        U_x = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_X_hb.mat')['ans']
        U_y = scipy.io.loadmat(f'MPCA/{dataset_exp[4:-3]}_Y_hb.mat')['ans']
        
    MSE_mpca = rmse_mpca(X_train_n, Y_train_n, X_test_n, Y_test_n, U_x, U_y, H_b[-1])
    with open(f"experiments/{dataset_exp}_hb_mpca.npz", "wb") as f:
        np.savez(f, **MSE_mpca)
    


def main_func(dataset_exp):
    dataset, exp = dataset_exp[:-3], dataset_exp[-2:]
    if dataset == 'ECoG32':
        X_test_n = np.transpose(scipy.io.loadmat('ECoG_X_test.mat')['X_hold_out'], (0,2,1))[1500:2250]
        Y_test_n = scipy.io.loadmat('ECoG_Y_test.mat')['Y_hold_out'][1500:2250]
        X_train_n = np.transpose(scipy.io.loadmat('ECoG_X_train.mat')['X_train'], (0,2,1))[:1500]
        Y_train_n = scipy.io.loadmat('ECoG_Y_train.mat')['Y_train'][:1500]
        
    elif dataset == 'ECoG64':
        X = scipy.io.loadmat('X_ECoG64.mat')['features1']
        Y = scipy.io.loadmat('Y_ECoG64.mat')['motion_dim1']

        X_test_n = np.transpose(X, (0,2,1))[3000:4500]
        Y_test_n = Y[3000:4500]
        X_train_n = np.transpose(X, (0,2,1))[:3000]
        Y_train_n = Y[:3000]
        
    elif dataset == 'ECoG86': 
        X = scipy.io.loadmat('X_ECoG86.mat')['features1']
        Y = scipy.io.loadmat('Y_ECoG86.mat')['motion_dim1']

        X_test_n = np.transpose(X, (0,2,1))[4000:6000]
        Y_test_n = Y[4000:6000]
        X_train_n = np.transpose(X, (0,2,1))[:4000]
        Y_train_n = Y[:4000]
    
    elif dataset == 'EEG60':
        X = scipy.io.loadmat('X_EEG60.mat')['features1']
        Y = scipy.io.loadmat('Y_EEG60.mat')['motion_dim1']

        X_test_n = np.transpose(X, (0,2,1))[2500:4250]
        Y_test_n = Y[2500:4250]
        X_train_n = np.transpose(X, (0,2,1))[:2500]
        Y_train_n = Y[:2500]
    
    X_test_n = (X_test_n - X_test_n.mean()) / X_test_n.std() 
    Y_test_n = (Y_test_n - Y_test_n.mean()) / Y_test_n.std()
    X_train_n = (X_train_n - X_train_n.mean()) / X_train_n.std()
    Y_train_n = (Y_train_n - Y_train_n.mean()) / Y_train_n.std()
    if exp == 'wt':
        wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n)
    elif exp == 'ht':
        ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n)
    elif exp == 'hs':
        hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n)
    elif exp == 'hb':
        hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n)

    


if __name__ == '__main__':
    de = [#'EEG60_wt', 'EEG60_ht', 'EEG60_hs', 'EEG60_hb','ECoG32_wt', 
          'ECoG32_ht', 'ECoG32_hs', 'ECoG32_hb',
          'ECoG64_wt', 'ECoG64_ht', 'ECoG64_hs', 'ECoG64_hb',
          #'EEG60_hs', 'EEG60_hb','ECoG86_wt', 'ECoG86_ht', 'ECoG86_hs', 'ECoG86_hb'
    ]
    
    p = multiprocessing.Pool(processes=4)
    
    for i,params in enumerate(de):
        print(params)
        p.apply_async(main_func, [params])
    p.close()
    p.join()