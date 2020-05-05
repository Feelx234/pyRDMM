import numpy as np


def getX(X_in,deg):
    if len(X_in.shape)==1:
        X_in=X_in[:,None]
    l=[np.power(X_in,i) for i in range(deg,0,-1)]
    l.append(np.ones(X_in.shape))
    return np.hstack(l)


def CooksDistance(beta1,beta2,XTX,s_2):
    assert(len(beta1)==len(beta2))   
    m=len(beta2)
    return (beta1-beta2).T @ XTX @ (beta1-beta2)/(m*s_2)


def getXTXforFit(X_in,Y_in,beta):
    N=len(X_in)
    m=len(beta)
    deg = len(beta)-1
    f=np.poly1d(beta)
    e=Y_in-f(X_in)
    X=getX(X_in,deg)
    
    s_2=np.sum(np.square(e))/(N-m)
    return X.T @ X, s_2




class Cov_Distance:
    def __init__(self, order):
        self.order = order

    def calculate_dataset_statistics(self, task, dataset_fit, model):
        self.XTX, self.s_2 = getXTXforFit(model.x, model.y, dataset_fit.beta)

    def compare(self, subgroup1, subgroup2, statistics1, statistics2):
        return CooksDistance(statistics2.beta, statistics2.beta, self.XTX, self.s_2)

    @property
    def requires_cover_arr(self):
        return False

    @property
    def is_symmetric(self):
        return True