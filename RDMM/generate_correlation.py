from collections import namedtuple
import numpy as np
import pandas as pd

def cov_to_corr(arr):
    d = arr.shape[0]
    arr_out=np.zeros(arr.shape)
    diags = [np.sqrt(np.abs(arr[i,i])) for i in range(d)]

    for i in range(d):
        for j in range(d):
            arr_out[i,j] = arr[i,j]/(diags[i]*diags[j])
    return arr_out


mean_cov_tuple=namedtuple('mean_cov_tuple',['mean', 'cov'])


def generate_cov_parameters(d, d_sigma=(0.1,10)):
    q, _ = np.linalg.qr(np.random.rand(d,d))
    arr=q @ np.diag(np.random.uniform(*d_sigma, size=d)) @ q.T
    return arr

def generate_all_cov_parameters(d, n_classes, min_dist=0.3, dist=np.linalg.norm):
    """ This funciton generates mean and covariance matrices
    """
    all_covs = []
    all_corrs = []
    while len(all_covs) < n_classes:
        new_cov = generate_cov_parameters(d)
        new_corr = cov_to_corr(new_cov)
        if all(dist(new_corr.flatten()-corr.flatten()) > min_dist*d for corr in all_corrs):
            all_covs.append(new_cov)
            all_corrs.append(new_corr)
    return [mean_cov_tuple(np.zeros(d), cov) for cov in all_covs]


def create_corr_samples(class_sizes,  parameters):
    assert(len(class_sizes) == len(parameters))
    samples=[]
    for (cls, sample_size), param in zip(enumerate(class_sizes), parameters):
        mean = param.mean
        cov = param.cov
        samples.append(np.random.multivariate_normal(mean, cov, size=sample_size))
    return samples

def create_cov_samples(class_sizes,  parameters):
    assert(len(class_sizes) == len(parameters))
    samples=[]
    for (cls, sample_size), param in zip(enumerate(class_sizes), parameters):
        mean = param.mean
        cov = param.cov
        samples.append(np.random.multivariate_normal(mean, cov, size=sample_size))
    return samples

def create_cov_dataframe(class_sizes, cov_matrices):
    samples = create_corr_samples(class_sizes, cov_matrices)
    class_array_list=[]
    for cls, sample in enumerate(samples):
        #print(sample.shape)
        class_array_list.append(np.full(sample.shape[0], cls, dtype=int))
    class_array = np.hstack(class_array_list)
    samples_stacked = np.vstack(samples)
    d={}
    print(class_array.shape)
    print(samples_stacked.shape)
    d["class"]=pd.Categorical(class_array)
    for i in range(samples_stacked.shape[1]):
        name = 'attr_{}'.format(str(i))
        d[name] = samples_stacked[:,i].flatten()
    return pd.DataFrame.from_dict(d)