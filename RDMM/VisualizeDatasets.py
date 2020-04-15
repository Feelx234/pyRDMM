import seaborn
import numpy as np
import matplotlib.pyplot as plt
from .CreateDataSets import getBivariateGaussianAlongLine

from itertools import repeat
def visualize_fits(df,fits):
    dx=50
    dy=50
    #fig=plt.figure()
    x=np.linspace(-dx,dx,30)
    uniques=np.unique(df['class'])
    n_unique=len(uniques)
    palette = seaborn.color_palette("hls", n_unique)
    for sel,a,b in fits:
        if sel.attribute_name.startswith("Noise"):
            seaborn.lineplot(x,x*a+b,hue=np.zeros(x.shape), palette=[palette[0]],legend=None)
    visualizeArtificialDataFrame(df, palette)
    plotfit((df['x'],df['y']),'k',dx)
    n=1
    for i,(sel,a,b) in enumerate(fits):
        if sel.attribute_name.startswith("Noise"):
            #plt.plot(x,x*a+b,'k')
            pass
        else:
            seaborn.lineplot(x,x*a+b,hue=np.zeros(x.shape), palette=[palette[n]], legend=None)
            n+=1
            #plt.plot(x,x*a+b)

    plt.xlim(-dx, dx)
    plt.ylim(-dy, dy)

def visualizeArtificialDataFrame(df, palette=None):

    #fg = seaborn.FacetGrid(data=df, hue='class', aspect=1.61)
    #fg.map(plt.scatter, 'x', 'y').add_legend()
    #viz_df=pd.DataFrame.from_dict({'x':df['x'],'y':df['y'],'class':df['class'].apply(str)})
    uniques=np.unique(df['class'])
    n_unique=len(uniques)
    if palette is None:
        seaborn.color_palette("hls", n_unique)
    seaborn.scatterplot('x','y',hue='class',data=df,palette=palette,linewidth=0)


def plotfit(points,color,dx):
    x=np.linspace(-dx,dx)
    if isinstance(points, tuple):
        x_fit,y_fit=points
    else:
        x_fit,y_fit=points.T
    res=np.polyfit(x_fit,y_fit,1)
    y2=res[1]+res[0]*x
    return plt.plot(x,y2,color=color,linestyle='--')

def plotLineAndGaussianAndFit(n,slope,intersept,sigma1,sigma2,x_center,color,dx=30,plot_fit=True,plot_line=True):
    axes=[]
    if plot_line:
        x=np.linspace(-dx,dx)
        y=slope*x+intersept
        axes.append(plt.plot(x,y,color=color)[0])
    points=getBivariateGaussianAlongLine(n,slope,intersept,sigma1,sigma2,x_center)
    axes.append(plt.scatter(points[:,0],points[:,1],color=color))
    if plot_fit:
        axes.append(plotfit(points,color,dx)[0])
    return points,tuple(axes)