import numpy as np
import torch
from tensorly.tenalg import mode_dot


class hankelization:
    """
    Class for hanke
    """
    def __init__(self, Ls, dims):
        self.dims = dims
        self.Ls = Ls
        
    def forward(self, X):
        self.S = []
        for l,d in zip(self.Ls,self.dims):
            s = np.zeros((l*(X.shape[d]-l+1),X.shape[d]))
            for i in range(X.shape[d]-l+1):
                s[i*l:(i+1)*l,i:i+l] = np.eye(l)
            self.S.append(s)
        X_new = X.copy()
        for s,d in zip(self.S, self.dims):
            X_new = mode_dot(X_new,s,d)
        self.new_shape = X_new.shape
        shape_new = []
        order_new = []
        i = 0
        for d in range(len(X.shape)):
            if d in self.dims:
                l = self.Ls[self.dims.index(d)]
                shape_new.append(l)
                shape_new.append((X.shape[d]-l+1))
                order_new.extend([i+1,i])
                i+=2
            else:
                shape_new.append(X.shape[d])
                order_new.append(i)
                i+=1
        self.order_new = order_new
        X_new = X_new.reshape(shape_new)
        X_new = np.transpose(X_new, order_new)
        self.S_back = [np.linalg.pinv(s) for s in self.S]
        return X_new
    
    def back(self, X):
        X_new = X.copy()
        X_new = np.transpose(X_new, self.order_new)
        X_new = X_new.reshape(self.new_shape)
        for s,d in zip(self.S_back, self.dims):
            X_new = mode_dot(X_new,s,d)
        return X_new
    
    
# def hankel_matrix(tensor, dim, L):
#     tensor_shape = tensor.shape
#     perm, unfolding = unfold(tensor, dim)
#     matrix = np.zeros((tensor_shape[dim]-L+1,L,np.prod([tensor_shape[i] for i in perm[1:]])))
#     for i in range(unfolding.shape[1]):
#         matrix[:,:,i] =  hankel(unfolding[:L,i],unfolding[L-1:,i]).T
#     matrix = matrix.reshape([tensor_shape[dim]-L+1,L]+[tensor_shape[i] for i in perm[1:]])
#     perm = np.array(perm)
#     matrix = np.transpose(matrix, [2+i for i in range(sum(perm<dim))]+[0,1]+[len(tensor_shape)-i for i in range(sum(perm>dim))][::-1])
#     return matrix


# def from_hankel_matrix(matrix,dim):
#     shape_m = matrix.shape
#     L = shape_m[dim]+shape_m[dim+1]-1
#     D = len(shape_m)
#     perm = [dim, dim+1]
#     for i in range(D):
#         if i not in [dim, dim+1]:
#             perm.append(i)
#     matrix = np.transpose(matrix,perm)
#     matrix = matrix.reshape([shape_m[i] for i in perm])
#     matrix = np.concatenate((matrix[0],matrix[:,-1][1:]),axis=0)
#     perm = np.array(perm)
#     matrix = np.transpose(matrix, [1+i for i in range(sum(perm<dim))]+[0]+[len(shape_m)-2-i for i in range(sum(perm>dim+1))][::-1])
#     return matrix
            
        