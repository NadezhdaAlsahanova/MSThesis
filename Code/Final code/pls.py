"""
File for finding the best parameters for HOPLS model
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


params = {'ECoG32': {'X_wt': [6,8], 'Y_wt': [3], 'X_ht': [4,4,4], 'Y_ht': [4,3], 'X_hs': [5,5,3], 'Y_hs': [3], 'X_hb': [4,4,4,3], 'Y_hb': [4,3],},
         'ECoG64': {'X_wt': [7,8], 'Y_wt': [3], 'X_ht': [6,4,5], 'Y_ht': [4,3], 'X_hs': [6,5,3], 'Y_hs': [3], 'X_hb': [4,3,4,3], 'Y_hb': [4,3],},
         'ECoG86': {'X_wt': [6,8], 'Y_wt': [4], 'X_ht': [4,4,4], 'Y_ht': [4,4], 'X_hs': [5,5,3], 'Y_hs': [3], 'X_hb': [4,4,4,4], 'Y_hb': [4,4],},
         'EEG60': {'X_wt': [4,9], 'Y_wt': [5], 'X_ht': [5,4,9], 'Y_ht': [4,3], 'X_hs': [5,6,4], 'Y_hs': [3], 'X_hb': [3,4,7,3], 'Y_hb': [4,5],},}


def wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, X_sh, Y_sh):    
    MSE_test = {}
    for ks1 in range(1,X_sh[0]):
        print('Starts for ks1 =', ks1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for ks2 in range(1,X_sh[1]):
            for ks3 in range(1,Y_sh[0]):
                Ln = [ks1,ks2]
                Km = [ks3]
                rmse_test, rmse_train, R = compute_rmse_hopls(X_train_n, Y_train_n, X_test_n, Y_test_n, Ln, Km, R_max=30)
                MSE_test[f'Ln = ({ks1},{ks2}), Km = {ks3}, R = {R}'] = rmse_test
                
        with open(f"experiments/{dataset_exp}_hopls.npz", "wb") as f:
            np.savez(f, **MSE_test)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')

    
def ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, X_sh, Y_sh):
    data, H_t = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10],[10]], [[0],[0]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data

    MSE_test = {}
    for ks1 in range(1,X_sh[0]):
        print('Starts for ks1 =', ks1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for ks2 in range(1,X_sh[1]):
            for ks3 in range(1,X_sh[2]):
                for ks4 in range(1,Y_sh[0]):
                    for ks5 in range(1,Y_sh[1]):
                        Ln = [ks1,ks2,ks3]
                        Km = [ks4,ks5]
                        rmse_test, rmse_train, R = compute_rmse_hopls(X_train_n, Y_train_n, X_test_n, Y_test_n, Ln, Km, R_max=15, hankelization=[H_t[-2],H_t[-1]])
                        MSE_test[f'Ln = ({ks1},{ks2},{ks3}), Km = ({ks4},{ks5}), R = {R}'] = rmse_test

        with open(f"experiments/{dataset_exp}_hopls.npz", "wb") as f:
            np.savez(f, **MSE_test)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
    

def hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, X_sh, Y_sh):  
    data, H_s = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[2],[]], [[2],[]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data

    MSE_test = {}
    for ks1 in range(1,X_sh[0]):
        print('Starts for ks1 =', ks1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for ks2 in range(1,X_sh[1]):
            for ks3 in range(1,X_sh[2]):
                for ks4 in range(1,Y_sh[0]):
                    Ln = [ks1,ks2,ks3]
                    Km = [ks4]
                    rmse_test, rmse_train, R = compute_rmse_hopls(X_train_n, Y_train_n, X_test_n, Y_test_n, Ln, Km, R_max=15)
                    MSE_test[f'Ln = ({ks1},{ks2},{ks3}), Km = ({ks4}), R = {R}'] = rmse_test

        with open(f"experiments/{dataset_exp}_hopls.npz", "wb") as f:
            np.savez(f, **MSE_test)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')


    
def hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, X_sh, Y_sh):
    data, H_b = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10,2],[10]], [[0,2],[0]])
    X_train_n, Y_train_n, X_test_n, Y_test_n = data

    MSE_test = {}
    for ks1 in range(1,X_sh[0]):
        print('Starts for ks1 =', ks1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for ks2 in range(1,X_sh[1]):
            for ks3 in range(1,X_sh[2]):
                for ks4 in range(1,X_sh[3]):
                    for ks5 in range(1,Y_sh[0]):
                         for ks6 in range(1,Y_sh[1]):
                            Ln = [ks1,ks2,ks3,ks4]
                            Km = [ks5,ks6]
                            rmse_test, rmse_train, R = compute_rmse_hopls(X_train_n, Y_train_n, X_test_n, Y_test_n, Ln, Km, R_max=15, hankelization=[H_b[-2],H_b[-1]])
                            MSE_test[f'Ln = ({ks1},{ks2},{ks3},{ks4}), Km = ({ks5},{ks6}), R = {R}'] = rmse_test

        with open(f"experiments/{dataset_exp}_hopls.npz", "wb") as f:
            np.savez(f, **MSE_test)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')


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
    train_params = params[dataset]
    if exp == 'wt':
        wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['X_wt'], train_params['Y_wt'])
    elif exp == 'ht':
        ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['X_ht'], train_params['Y_ht'])
    elif exp == 'hs':
        hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['X_hs'], train_params['Y_hs'])
    elif exp == 'hb':
        hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['X_hb'], train_params['Y_hb'])
        
        
if __name__ == '__main__':
    de = ['EEG60_ht', 'EEG60_hs',# 'EEG60_wt', 'EEG60_hb','ECoG32_wt', 
          #'ECoG32_ht', 'ECoG32_hs', 'ECoG32_hb','ECoG64_wt', 
        'EEG60_hb', 'ECoG64_hs', 'ECoG64_hb',
          #'EEG60_hs', 'ECoG86_wt', 'ECoG86_ht', 'ECoG86_hs', 'ECoG86_hb'
    ]
    
    p = multiprocessing.Pool(processes=2)
    
    for i,params in enumerate(de):
        print(params)
        p.apply_async(main_func, [params])
    p.close()
    p.join()