import numpy as np
import pysubgroup as ps
#import seaborn
#import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from collections import namedtuple

from .generate_correlation import generate_all_cov_parameters, create_cov_dataframe

def getGaussParamsForLine(slope,intersept,sigma1,sigma2,x_center):
    phi=np.arctan(slope)
    sin_phi=np.sin(phi)
    cos_phi=np.cos(phi)
    rot=np.array([[cos_phi,-sin_phi],[sin_phi,cos_phi]])
    diag=np.array([[sigma1*sigma1, 0],[0, sigma2*sigma2]])
    cov=np.matmul(np.matmul(rot,diag),rot.T)
    mean=np.array([x_center,x_center*slope+intersept])
    return mean,cov

def getBivariateGaussianAlongLine(n,slope,intersept,sigma1,sigma2,x_center):
    mean,cov=getGaussParamsForLine(slope,intersept,sigma1,sigma2,x_center)
    return np.random.multivariate_normal(mean,cov,n)







def generateParametersForLines(n_injections,limits):
    #slope,intersept,sigma1,sigma2,x_center
    params=np.array([np.random.rand(n_injections)*(limits[i,1]-limits[i,0])+limits[i,0] for i in range(n_injections)])
    return params.T

def generateParametersForLines2(n_injections,limits):
    #slope,intersept,sigma1,sigma2,x_center
    params=np.array([np.random.permutation(np.linspace(limits[i,0],limits[i,1],n_injections)) for i in range(len(limits))])
    return params.T



def classesAndPointsToDataFrame(classes,points,names=['x','y']):
    d={}
    x,y=points
    d["class"]=pd.Categorical(classes)
    d[names[0]]=x
    d[names[1]]=y
    return pd.DataFrame.from_dict(d)



def generateDataFrameForLinesByInjection(ns,line_params):
    #the first entry in params/ns is assumed to provide the trend
    points=[]
    labels=[]
    for i,(curr_n,params) in enumerate(zip(ns,line_params)):
        mean,cov=getGaussParamsForLine(*params)
        points.append(np.random.multivariate_normal(mean,cov,curr_n))
        labels.append(np.full(curr_n,i,dtype=int))
    points=np.vstack(points)
    return classesAndPointsToDataFrame(np.hstack(labels),points.T)


def generateDataFrameForLinesByInjection_uncorrelated(ns,line_params):
    #the first entry in params/ns is assumed to provide the trend
    points=[]
    labels=[]
    for i,(curr_n,params) in enumerate(zip(ns,line_params)):
        x_center,y0,slope,sigma1,sigma2,_= params
        x = sigma1 * np.random.randn(curr_n,1) + x_center
        y = slope * (x - x_center) + y0 + sigma2 * np.random.randn(curr_n,1)
        xy=np.hstack([x,y])
        points.append(xy)
        labels.append(np.full(curr_n,i,dtype=int))
    points=np.vstack(points)
    return classesAndPointsToDataFrame(np.hstack(labels),points.T)



def addNoiseParams(df, noise_fractions, n=-1):
    if n==-1:
        n=len(df.index)
    for i, p in enumerate(noise_fractions):
        arr = np.zeros(len(df), dtype = bool)
        n_True = int(p * len(df))
        arr[0:n_True] = True
        np.random.shuffle(arr)
        df["Noise_" + str(i)] = arr
    return df

def add_noise_to_dataframe(df, num_noise_attributes):
    _, counts = np.unique(df['class'], return_counts=True)
    min_counts = np.min(counts)

    min_noise_fraction = min_counts/len(df)
    df = addNoiseParams(df, np.random.uniform(min_noise_fraction, 0.9, num_noise_attributes))
    return df



def generateProbabilities(line_params,weights,points):
    probabilities = np.empty((len(points),len(line_params)))
    for i,params in enumerate(line_params):
        mean,cov=getGaussParamsForLine(*params)
        probabilities[:,i]=weights[i]*multivariate_normal.pdf(points, mean=mean, cov=cov)
    return probabilities
    


from scipy.stats import multivariate_normal
def getClassesByLines_largest(line_params,weights,points):
    #the last entry in params/ns is assumed to provide the trend
    probabilities=generateProbabilities(line_params,weights,points)
    classes=np.argmax(probabilities, axis=1)
    return classes



def getClassesByLines_probability(line_params,weights,points,epsilon=1E-10):
    #the last entry in params/ns is assumed to provide the trend
    probabilities=generateProbabilities(line_params,weights,points)
    probabilities+=epsilon
    
    #probabilities= np.divide(probabilities,probabilities[:,-1])
    probabilities=(probabilities.T / np.sum(probabilities,axis=1)).T
    probabilities=np.cumsum(probabilities,axis=1)
    
    p=np.random.rand(len(points))
    nonzeros=np.count_nonzero(np.less(probabilities,p[:,None]),axis=1)

    classes=np.empty(len(points),dtype=int)
    for i in range(len(points)):
        classes[i]=nonzeros[i]

    
    
    return classes



def generateDataFrameForLinesByAssignment(n,line_params,weights,background="uniform",strategy="largest",epsilon=1E-10):
    #the first entry in params/ns is assumed to provide the trend
    # assigment=[largest, distriution]
    
    
    if not (len(line_params)==len(weights)):
        raise ValueError("line_params and weights need to be of the same length")
        
    
    points=[]
    if not isinstance(background,str):
        points=background
    else:
        
        if background=="normal":
            mean,cov=getGaussParamsForLine(*line_params[0,:])
            points=np.random.multivariate_normal(mean,cov,n)
        elif background=="uniform":
            points=np.array([np.random.uniform(-30,30,n),np.random.uniform(-50,50,n)]).T
        else:
            raise ValueError
        
    if strategy=="distribution":
        classes=getClassesByLines_probability(line_params,weights,points,epsilon)
    elif strategy=="largest":
        classes=getClassesByLines_largest(line_params,weights,points)
    return classesAndPointsToDataFrame(classes,points.T)





def genNsFromAverage(averages,relative_variation=0.20):
    result=np.empty(len(averages),dtype=int)
    for i,target in enumerate(averages):
        d=target*relative_variation
        result[i]=np.random.randint(target-d,target+d)
    return result


def removePointsOutsideOfBox(df,selectors):
    for sel in selectors:
        df=df[sel.covers(df)]
    df.reset_index(drop=True,inplace=True)
    return df




def getTwoDataFrames(n_targets,n_noise=2000,noise_attributes=10, noise_params=[0,0,10,20,0],limits=np.array([[-5,5],[-10,10],[5,15],[1,5],[-5,5]]),is_correlated=False,
                     InlierSelectors=[ps.IntervalSelector('x',-40,40),ps.IntervalSelector('y',-40,40)]):
    
    params=generateParametersForLines2(n_targets,limits)

    params=np.vstack([noise_params,params])
    

    def getDataFrame():
        ns_limits=np.hstack([[n_noise],np.random.randint(200,300,len(params))] )
        ns=genNsFromAverage(ns_limits)
        df=generateDataFrameForLinesByInjection(ns,params) if is_correlated else generateDataFrameForLinesByInjection_uncorrelated(ns,params)
        df=removePointsOutsideOfBox(df,InlierSelectors)
        df=addNoiseParams(df,np.random.uniform(0.2,0.8,noise_attributes))
        return df

    df1=getDataFrame()
    df2=getDataFrame()

    return df1, df2


# regression dataframe methods


regression_tpl = namedtuple('regression_tpl',['x0','y0','slope','sigmax','sigmay','phi'])

def create_phi(d_phi, n):
    phi =np.random.uniform(d_phi,np.pi - d_phi,n)
    phi *= 2 * (np.random.rand(n) > 0.5) - 1
    phi += np.pi/2
    return phi

def generate_regression_parameters(n, phi_noise = 0, dy0 = 30, dx0 = 30, max_x = 50, d_phi = 10 / 360 * np.pi):
    y0 = np.random.uniform(-dy0, dy0, n)
    x0 = np.random.uniform(-dx0, dx0, n)
    phi = create_phi(d_phi, n)
    slope = np.tan(phi)
    sigma_x = np.random.uniform(2, (max_x - abs(x0))/3)
    sigma_y = np.random.uniform(sigma_x/20, sigma_x/10)
    return [regression_tpl(*tpl) for tpl in zip(x0, y0, slope, sigma_x, sigma_y, phi)]


def generate_all_regression_parameters(n_classes):
    d_phi_noise = 20 / 360 * np.pi
    phi_noise = np.random.uniform(-d_phi_noise, d_phi_noise)
    noise_slope = np.tan(phi_noise)
    noise_params = regression_tpl(0, 0, noise_slope, 50/2.2, 50/2.2, phi_noise)

    class_parameters = generate_regression_parameters(n_classes, phi_noise)
    all_parameters = [noise_params] + class_parameters
    return all_parameters


def generate_regression_dataframe(sample_sizes, parameters, InlierSelectors=[ps.IntervalSelector('x',-50,50),ps.IntervalSelector('y',-50,50)]):
    df = generateDataFrameForLinesByInjection_uncorrelated(sample_sizes, parameters)
    df = removePointsOutsideOfBox(df, InlierSelectors)
    return df


def generate_two_regression_dataframes(background_sizes, n_classes, num_noise_attributes):
    return generate_two_dataframes(background_sizes, n_classes, num_noise_attributes, generate_regression_dataframe, generate_all_regression_parameters)



# Generic dataframe generation methods

def generate_sample_sizes(background_size, num_classes):
    min_size = 0.05 * background_size
    max_size = 0.1 * background_size
    sample_sizes = [background_size] + list(np.random.randint(min_size, max_size + 1, num_classes - 1))
    return sample_sizes

def get_dataframe_from_params(sample_func, sample_sizes, parameters, num_noise_attributes):
    df = sample_func(sample_sizes, parameters)
    df = add_noise_to_dataframe(df, num_noise_attributes)
    return df

def generate_two_dataframes(background_sizes, n_classes, num_noise_attributes, sample_func, params_func):
    sample_sizes1 = generate_sample_sizes(background_sizes[0], n_classes)
    sample_sizes2 = generate_sample_sizes(background_sizes[1], n_classes)
    all_parameters = params_func(n_classes)

    df1 = get_dataframe_from_params(sample_func, sample_sizes1, all_parameters, num_noise_attributes)
    df2 = get_dataframe_from_params(sample_func, sample_sizes2, all_parameters, num_noise_attributes)

    return df1, df2, all_parameters, sample_sizes1, sample_sizes2



def doFitsForLiterals(df,Literals):
    sels=chain.from_iterable([lit.selectors for lit in Literals])
    fits=[]
    for sel in sels:
        
        df_sels=df[sel.covers(df)]
        fits.append([sel,*np.polyfit(df_sels["x"],df_sels["y"],1)])

    return fits



def doFitsForDataFrame(df):
    columns=[x for x in df.columns if (x not in ['x','y'])]
    Literals=SelectorGenerator().convertLiteralStringsToLiterals(df,columns)
    fits=doFitsForLiterals  (df,Literals)
    return df,fits



import itertools
def hide(df_in,depth,n_noise): # has test
    if depth<=1:
        raise ValueError("The depth parameter is expected to be larger than one.")
    if (n_noise/depth) != (n_noise//depth):
        raise ValueError("The number of noise points has to be divisible by the depth")
    df=df_in.copy()
    classes=np.unique(df['class'])
    for c in classes:
        if c>0: # don't hide background class
            class_cover=np.array(df['class']==c)
            choices=np.nonzero(np.logical_not(class_cover))[0]
            np.random.shuffle(choices)
            choices=np.split(choices[:n_noise],depth)
            for d,subset in enumerate(itertools.combinations(choices, len(choices)-1)):
                hidden_cover=class_cover.copy()
                hidden_cover[np.hstack(subset)]=True
                df['class_{}_{}'.format(c,d)]=hidden_cover
    return df


from collections import defaultdict
def validate_hide(df): # is test
    classes=np.unique(df['class'])
    d=defaultdict(list)
    
    for col in df.columns:
        if col.startswith('class_'):
            split_result=str.split(col,'_')
            d[int(split_result[1])].append(df[col])
    for c in classes:
        if c>0:
            arrs=d[c]
            target=df['class']==c
            
            for L in range(1, len(arrs)+1):
                for subset in itertools.combinations(arrs, L):
                    alls=np.all(subset,axis=0)
                    if L < len(arrs):
                        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, target,alls )
                    else:
                        np.testing.assert_array_equal(target,alls)
            print("test for class {} succeeded".format(c))

def randomize_class_sizes(class_sizes, delta):
    arr_sizes = np.array(class_sizes)
    min_sizes =  arr_sizes - delta * arr_sizes
    max_sizes =  arr_sizes + delta * arr_sizes
    return  [np.random.randint(mi, ma) for mi, ma in zip(min_sizes, max_sizes)]


from collections import Counter
from itertools import repeat


######         --- begin transition ---


def create_random_transition_matrix(n_states): # has test
    while True:
        Trans=np.random.rand(n_states, n_states)
        vals=np.sum(Trans,axis=1)
        for i in range(n_states):
            Trans[i,:]/=vals[i]
        vals, vecs=np.linalg.eig(Trans.T)
        if np.all(np.isreal(vals)) and (np.all(vecs[:,0]>=0) or np.all(vecs[:,0]<=0)) and np.isclose(vals[0],1) and np.all(vals[1:]<0.5):
            break
    P=np.empty((n_states,n_states))
    eig_v=vecs[:,0]/np.sum(vecs[:,0])
    for i in range(n_states):
        P[i,:] = eig_v[i]* Trans[i,:]
    return P

def create_transition_sample(n_states, sample_size, pmatrix): # has test
        choices = np.random.choice(n_states**2, sample_size, p=pmatrix.flatten())
        ins, outs = np.unravel_index(choices, pmatrix.shape) # pylint: disable=unbalanced-tuple-unpacking
        return ins, outs

def create_transition_samples(class_sizes,  transition_matrices, starting_ps=None):
    n_states = transition_matrices[0].shape[0]
    classes=[]
    all_ins=[]
    all_outs=[]
    for (cls, sample_size), pmatrix in zip(enumerate(class_sizes),transition_matrices):
        ins, outs = create_transition_sample(n_states, sample_size, pmatrix)
        classes.append(cls)
        all_ins.append(ins)
        all_outs.append(outs)
    return classes, all_ins, all_outs


def create_transition_dataframe(class_sizes,transition_matrices):
    classes,all_ins,all_outs = create_transition_samples(class_sizes, transition_matrices)
    class_array_list=[]
    for cls,size in zip(classes,map(len,all_ins)):
        class_array_list.append(np.full(size, cls, dtype=int))
    class_array = np.hstack(class_array_list)
    ins = np.hstack(all_ins)
    outs = np.hstack(all_outs)
    return classesAndPointsToDataFrame(class_array, (ins,outs), ['in','out'])

def create_transition_params(n_states, n_classes):
    return [create_random_transition_matrix(n_states) for _ in range(n_classes + 1)]

import functools

def generate_two_transition_dataframes(background_sizes, n_classes, num_noise_attributes, n_states): # has test
    params_func = functools.partial(create_transition_params, n_states)
    return generate_two_dataframes(background_sizes, n_classes, num_noise_attributes, create_transition_dataframe, params_func)


######         --- end transition ---


def Zipfs_datasets(n_words,):
    pass

def create_Zipfs_distribution(n_words, n_swaps, initial_fuel = 10):
    #word_order = np.arange(n_words)
    words_to_swap = np.random.random_integers(int(initial_fuel**1.5), n_words, size=n_swaps)
    new_positions = []
    for word_id in words_to_swap:
        fuel = initial_fuel
        curr_id=word_id
        while fuel > 0:
            fuel -= (1/curr_id - 1/word_id)
            curr_id -= 1
        new_positions.append((curr_id, word_id))
    print(new_positions)


######         --- begin covariance ---    


def generate_two_cov_dataframes(background_sizes, n_classes, num_noise_attributes, n_states): # has test
    params_func = functools.partial(generate_all_cov_parameters, n_states)
    return generate_two_dataframes(background_sizes, n_classes, num_noise_attributes, create_cov_dataframe, params_func)

