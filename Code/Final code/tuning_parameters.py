"""
File for finding the best parameters for Autoencoder  model
"""

import os
seed = 50
os.environ['PYTHONHASHSEED']=str(seed)
import scipy.io
import time
# import seaborn as sns
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
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from tensorly.regression.tucker_regression import TuckerRegressor
from hopls import matricize, qsquared, HOPLS
import random
from qpfs import *
from Autoencoder import *
from utils import *
from hankelization import hankelization
import multiprocessing
random_seed = 5
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
# !CUBLAS_WORKSPACE_CONFIG=:16:8#:4096:8
#device='cuda:0'
device='cuda:0'

lr = 0.01
batch_size = 512
loss_fn = my_loss(3)
epochs = 40
step_size = 100

params = {'ECoG32': {'t_N': 8, 't_K': 14, 'ht_K1': 8, 'ht_K2': 8, 'ht_N': 5,
                    'hs_K1': 13, 'hs_K2': 3, 'hs_N': 7, 'hb_K1': 6, 'hb_K2': 6, 'hb_K3': 3, 'hb_N': 4},
         'ECoG64': {'t_N': 11, 't_K': 14, 'ht_K1': 8, 'ht_K2': 8, 'ht_N': 5,
                    'hs_K1': 13, 'hs_K2': 3, 'hs_N': 7, 'hb_K1': 6, 'hb_K2': 6, 'hb_K3': 3, 'hb_N': 4},
         'ECoG86': {'t_N': 12, 't_K': 14, 'ht_K1': 8, 'ht_K2': 8, 'ht_N': 5,
                    'hs_K1': 13, 'hs_K2': 3, 'hs_N': 7, 'hb_K1': 6, 'hb_K2': 6, 'hb_K3': 3, 'hb_N': 4},
         'EEG60': {'t_N': 11, 't_K': 14, 'ht_K1': 8, 'ht_K2': 8, 'ht_N': 5,
                    'hs_K1': 13, 'hs_K2': 3, 'hs_N': 7, 'hb_K1': 6, 'hb_K2': 6, 'hb_K3': 3, 'hb_N': 4},}


def wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, N_max, K_max):
    X_test = torch.from_numpy(X_test_n).float().to(device)
    Y_test = torch.from_numpy(Y_test_n).float().to(device)
    X_train = torch.from_numpy(X_train_n).float().to(device) 
    Y_train = torch.from_numpy(Y_train_n).float().to(device) 
    fr, ch, y_sh = X_test_n.shape[1], X_test_n.shape[2], Y_test_n.shape[1]

    Parameters = {}
    RMSE_t = {}
    for K in range(2,K_max,2):
        print('Starts for k =', K, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for N in range(2,N_max,2):        
            if K > 16:
                ks = [fr,32,K]
            elif K > 8:
                ks = [fr,32,16,K]
            else:
                ks = [fr,32,16,8,K]
            model = AutoEncoder_tensor(3, [ks], [ch,N], y_sh, indexing=False).to(device)
            model = train(model, X_train, Y_train, X_test, Y_test, lr, step_size, batch_size, epochs, loss_fn, False)

            X_train_lower = model.encode(X_train).cpu().detach().numpy()
            X_test_lower = model.encode(X_test).cpu().detach().numpy()

            RMSE_dict = {}
            for rank1 in range(1,X_train_lower.shape[1]+1,1):   
                for rank2 in range(1,X_train_lower.shape[2]+1,1):
                    try:
                        RMSE_dict[f'{rank1},{rank2}'] = tensor_regression(X_train_lower, Y_train_n, X_test_lower, Y_test_n, [rank1,rank2])
                    except:
                        continue

            # Parameters[f'{K},{N}'] = sum(p.numel() for p in model.parameters())
            RMSE_t[f'{K},{N}'] = RMSE_dict[min(RMSE_dict, key=RMSE_dict.get)]
            
        with open(f"experiments/{dataset_exp}.npz", "wb") as f:
            np.savez(f, **RMSE_t)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')

    
def ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, N_max, K1_max, K2_max):
    fr, ch, y_sh = X_test_n.shape[1], X_test_n.shape[2], Y_test_n.shape[1]
    data, H_t = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10],[10]], [[0],[0]])
    X_tr_ht_n, Y_tr_ht_n, X_te_ht_n, Y_te_ht_n = data
    
    X_test = torch.from_numpy(X_te_ht_n).float().to(device)
    Y_test = torch.from_numpy(Y_te_ht_n).float().to(device)
    X_train = torch.from_numpy(X_tr_ht_n).float().to(device) 
    Y_train = torch.from_numpy(Y_tr_ht_n).float().to(device) 

    Parameters = {}
    RMSE_ht_t = {}

    for K1 in range(1,K1_max,2):
        print('Starts for k =', K1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for K2 in range(1,K2_max,2):
            for N in range(1,N_max):
                if K1 > 14:
                    ks1 = [fr,32,K1]
                elif K1 > 7:
                    ks1 = [fr,32,16,K1]
                else:
                    ks1 = [fr,32,16,8,K1]

                if K2 > 16:
                    ks2 = [ch,K2]
                elif K2 > 8:
                    ks2 = [ch,16,K2]
                else:
                    ks2 = [ch,16,8,K2]

                model = AutoEncoder_tensor(4, [ks1, ks2], [10,N], y_sh*10, indexing=True).to(device)
                model = train(model, X_train, Y_train, X_test, Y_test, lr, step_size, batch_size, epochs, loss_fn,False)

                torch.cuda.empty_cache()
                X_train_lower = model.encode(X_train).cpu().detach().numpy()
                X_test_lower = model.encode(X_test).cpu().detach().numpy()

                RMSE_dict = {}

                for rank1 in range(1,X_train_lower.shape[1]+1,int(X_train_lower.shape[1]/5)+1):   
                    for rank2 in range(1,X_train_lower.shape[2]+1,int(X_train_lower.shape[2]/5)+1):
                        for rank3 in range(1,X_train_lower.shape[3]+1,int(X_train_lower.shape[3]/5)+1):
                                RMSE_dict[f'{rank1},{rank2},{rank3}'] = tensor_regression(X_train_lower, Y_tr_ht_n, X_test_lower, Y_te_ht_n, [rank1,rank2,rank3], H_t[-1])


                # Parameters[f'{N},{K1},{K2}'] = sum(p.numel() for p in model.parameters())
                print(f'{dataset_exp}, Finished tensor regression {K1},{N},{K2}')
                RMSE_ht_t[f'{N},{K1},{K2}'] = RMSE_dict[min(RMSE_dict, key=RMSE_dict.get)]
                
        with open(f"experiments/{dataset_exp}.npz", "wb") as f:
            np.savez(f, **RMSE_ht_t)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
    

def hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, N_max, K1_max, K2_max):  
    fr, ch, y_sh = X_test_n.shape[1], X_test_n.shape[2], Y_test_n.shape[1]
    
    data, H_s = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[2],[]], [[2],[]])
    X_tr_hs_n, Y_tr_hs_n, X_te_hs_n, Y_te_hs_n = data
    X_tr_hs_n, X_te_hs_n = X_tr_hs_n.transpose(0,2,1,3), X_te_hs_n.transpose(0,2,1,3)
    X_test = torch.from_numpy(X_te_hs_n).float().to(device)
    Y_test = torch.from_numpy(Y_te_hs_n).float().to(device)
    X_train = torch.from_numpy(X_tr_hs_n).float().to(device) 
    Y_train = torch.from_numpy(Y_tr_hs_n).float().to(device) 

    Parameters = {}
    RMSE_hs_t = {}

    for K1 in range(1,K1_max,2):
        print('Starts for k =', K1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for K2 in range(1,K2_max):
            for N in range(1,N_max,2):
                if K1 > 14:
                    ks1 = [fr,32,K1]
                elif K1 > 7:
                    ks1 = [fr,32,16,K1]
                else:
                    ks1 = [fr,32,16,8,K1]

                model = AutoEncoder_tensor(4, [ks1, [2,3,K2]], [ch-1,N], y_sh, indexing=True).to(device)
                model = train(model, X_train, Y_train, X_test, Y_test, lr, step_size, batch_size, epochs, loss_fn, False)

                torch.cuda.empty_cache()
                X_train_lower = model.encode(X_train).cpu().detach().numpy()
                X_test_lower = model.encode(X_test).cpu().detach().numpy()
                
                RMSE_dict = {}

                for rank1 in range(1,X_train_lower.shape[1]+1,int(X_train_lower.shape[1]/5)+1):   
                    for rank2 in range(1,X_train_lower.shape[2]+1,int(X_train_lower.shape[2]/5)+1):
                        for rank3 in range(1,X_train_lower.shape[3]+1,int(X_train_lower.shape[3]/5)+1):
                                RMSE_dict[f'{rank1},{rank2},{rank3}'] = tensor_regression(X_train_lower, Y_tr_hs_n, X_test_lower, Y_te_hs_n, [rank1,rank2,rank3], H_s[-1])
                print(f'{dataset_exp}, Finished tensor regression {K1},{N},{K2}')
                RMSE_hs_t[f'{K1},{N},{K2}'] = RMSE_dict[min(RMSE_dict, key=RMSE_dict.get)]
                
        with open(f"experiments/{dataset_exp}.npz", "wb") as f:
            np.savez(f, **RMSE_hs_t)
        
    print('Finished ', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')


    
def hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, N_max, K1_max, K2_max, K3_max):
    fr, ch, y_sh = X_test_n.shape[1], X_test_n.shape[2], Y_test_n.shape[1]
    data, H_b = tensorization(X_train_n, Y_train_n, X_test_n, Y_test_n, [[10,2],[10]], [[0,2],[0]])
    X_tr_hb_n, Y_tr_hb_n, X_te_hb_n, Y_te_hb_n = data

    X_test = torch.from_numpy(X_te_hb_n).float().to(device)
    Y_test = torch.from_numpy(Y_te_hb_n).float().to(device)
    X_train = torch.from_numpy(X_tr_hb_n).float().to(device) 
    Y_train = torch.from_numpy(Y_tr_hb_n).float().to(device) 

    Parameters = {}
    RMSE_hb_t = {}

    for K1 in range(1,K1_max,2):
        print('Starts for k =', K1, 'for', dataset_exp, f'at {time.strftime("%H:%M:%S", time.localtime())}')
        for K2 in range(1,K2_max,2):
            for K3 in range(1,K3_max):
                for N in range(1,N_max):
                    if K1 > 14:
                        ks1 = [fr,32,K1]
                    elif K1 > 7:
                        ks1 = [fr,32,16,K1]
                    else:
                        ks1 = [fr,32,16,8,K1]

                    if K2 > 16:
                        ks2 = [ch-1,K2]
                    elif K2 > 8:
                        ks2 = [ch-1,16,K2]
                    else:
                        ks2 = [ch-1,16,8,K2]

                    model = AutoEncoder_tensor(5, [ks1, ks2, [2,3,K3]], [10,N], y_sh*10, indexing=True).to(device)
                    model = train(model, X_train, Y_train, X_test, Y_test, lr, step_size, batch_size, epochs, loss_fn, False)

                    torch.cuda.empty_cache()
                    X_train_lower = model.encode(X_train).cpu().detach().numpy()
                    X_test_lower = model.encode(X_test).cpu().detach().numpy()

                    RMSE_dict = {}
                    
                    for rank1 in range(1,X_train_lower.shape[1]+1,int(X_train_lower.shape[1]/3)+1):   
                        for rank2 in range(1,X_train_lower.shape[2]+1,int(X_train_lower.shape[2]/3)+1):
                            for rank3 in range(1,X_train_lower.shape[3]+1,int(X_train_lower.shape[3]/3)+1):
                                for rank4 in range(1,X_train_lower.shape[4]+1,int(X_train_lower.shape[4]/3)+1):
                                    RMSE_dict[f'{rank1},{rank2},{rank3},{rank4}'] = tensor_regression(X_train_lower, Y_tr_hb_n, X_test_lower, Y_te_hb_n, [rank1,rank2,rank3,rank4], H_b[-1])
                    print(f'{dataset_exp}, Finished tensor regression {N},{K1},{K2},{K3}')
                    RMSE_hb_t[f'{N},{K1},{K2},{K3}'] = RMSE_dict[min(RMSE_dict, key=RMSE_dict.get)]
                    
        with open(f"experiments/{dataset_exp}.npz", "wb") as f:
            np.savez(f, **RMSE_hb_t)
        
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
        wt(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['t_N'], train_params['t_K'])
    elif exp == 'ht':
        ht(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['ht_N'], train_params['ht_K1'], train_params['ht_K2'])
    elif exp == 'hs':
        hs(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['hs_N'], train_params['hs_K1'], train_params['hs_K2'])
    elif exp == 'hb':
        hb(dataset_exp, X_test_n, Y_test_n, X_train_n, Y_train_n, train_params['hb_N'], train_params['hb_K1'], train_params['hb_K2'], train_params['hb_K3'])

        
if __name__ == '__main__':
    de = ['EEG60_ht', 'EEG60_hs', 'EEG60_hb', 'ECoG32_hb',          
          'ECoG64_ht', 'ECoG64_hs', 'ECoG64_hb',
          'ECoG86_ht', 'ECoG86_hs', 'ECoG86_hb']
    
    
    
#     p = multiprocessing.Pool(processes=4)
    
    for i,dataset_exp in enumerate(de):
        main_func(dataset_exp)
#         print(dataset_exp)
#         p.apply_async(main_func, [dataset_exp])
#     p.close()
#     p.join()