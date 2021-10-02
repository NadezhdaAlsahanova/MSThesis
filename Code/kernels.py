import numpy as np
from abc import abstractmethod, ABCMeta

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))


class Kernel(object):
    """
    Base, abstract kernel class
    """
    __metaclass__ = ABCMeta

    def __call__(self, data_1, data_2):
        return self._compute(data_1, data_2)

    @abstractmethod
    def _compute(self, data_1, data_2):
        """
        Main method which given two lists data_1 and data_2, with
        N and M elements respectively should return a kernel matrix
        of size N x M where K_{ij} = K(data_1_i, data_2_j)
        """
        raise NotImplementedError('This is an abstract class')

    def gram(self, data):
        """
        Returns a Gramian, kernel matrix of matrix and itself
        """
        return self._compute(data, data)

    @abstractmethod
    def dim(self):
        """
        Returns dimension of the feature space
        """
        raise NotImplementedError('This is an abstract class')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __mul__(self, value):
        if isinstance(value, Kernel):
            return KernelProduct(self, value)
        else:
            if isinstance(self, ScaledKernel):
                return ScaledKernel(self._kernel, self._scale * value)
            else:
                return ScaledKernel(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __div__(self, scale):
        return ScaledKernel(self, 1./scale)

    def __pow__(self, value):
        return KernelPower(self, value)


class Linear(Kernel):
    """
    Linear kernel, defined as a dot product between vectors
        K(x, y) = <x, y>
    """

    def __init__(self):
        self._dim = None

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return data_1.dot(data_2.T)

    def dim(self):
        return self._dim


class Polynomial(Kernel):
    """
    Polynomial kernel, defined as a power of an affine transformation
        K(x, y) = (a<x, y> + b)^p
    where:
        a = scale
        b = bias
        p = degree
    """

    def __init__(self, scale=1, bias=0, degree=2):
        self._dim = None
        self._scale = scale
        self._bias = bias
        self._degree = degree

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return (self._scale * data_1.dot(data_2.T) + self._bias) ** self._degree

    def dim(self):
        return self._dim ** self._degree

class RBF(Kernel):
    """
    Radial Basis Function kernel, defined as unnormalized Gaussian PDF
        K(x, y) = e^(-g||x - y||^2)
    where:
        g = gamma
    """

    def __init__(self, gamma=None):
        self._gamma = gamma

    def _compute(self, data_1, data_2):
        if self._gamma is None:
            # libSVM heuristics
            self._gamma = 1./data_1.shape[1]

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-self._gamma * dists_sq)

    def dim(self):
        return np.inf


class Exponential(Kernel):
    """
    Exponential kernel, 
        K(x, y) = e^(-||x - y||/(2*s^2))
    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = 2 * sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf

def centered_k(kernel):
    m = kernel.shape[0]
    return (np.eye(m)-np.ones((m,m))/m) @ kernel @ (np.eye(m)-np.ones((m,m))/m)