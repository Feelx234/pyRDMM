from functools import partial
from heapq import heappop
import pandas as pd
import numpy as np
from collections import defaultdict

def highlight_max(row,column_name):
    '''
    highlight the maximum in a Series yellow.
    '''
    l=[]
    selectors=row[column_name].Selectors()
    for sel in selectors:
        col=str(sel)
        if col.startswith('class'):
            if "==" in col:
                split_result=str.split(col,"==")
            else:
                split_result=str.split(col,'_')
            l.append(int(split_result[1]))
    val=False
    if len(l)==len(selectors):
        #print(l)
        val=np.all(np.array(l,dtype=int)==l[0])
    #print(val)
    if val:
        return ['background-color: green' for v in row]
    else:
        return ['' for v in row]

def display_result(rset_in,column_name='name'):
    rset=rset_in.copy()
    value=[]
    name=[]
    
    while len(rset)>0:
        v,n=heappop(rset)
        value.append(v)
        name.append(n)
    df=pd.DataFrame.from_dict({'value':value,'name':name})
    return df.style.apply(partial(highlight_max,column_name=column_name),axis=1)



def eval_framework_to_df(tuples):
    abc_tuples=[t[1][0:3] for t in tuples]
    #function_tuples=[t[1][3:5] for t in tuples]
    pd.MultiIndex.from_tuples(abc_tuples)
    d=defaultdict(list)
    for val,tpl in tuples:
        d[tuple(tpl[0:3])].append(tuple(tpl[3:]),val)
    for key in d:
        d[key]=[val for key, val in sorted(d[key])]
    Index = pd.MultiIndex.from_tuples(d.keys())
    return pd.DataFrame.from_records((d[key] for key in d), index = Index)