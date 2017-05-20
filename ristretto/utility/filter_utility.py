# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import matplotlib.mlab as mlab

def vis_square(data):
    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
   
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data.squeeze()); 
    plt.axis('off')
    plt.show()

def weights_hist(weights):
    fig = plt.figure()
    results, edges = np.histogram(weights, normed=True)
    binWidth = edges[1] - edges[0]
    plt.bar(edges[:-1], results*binWidth, binWidth)
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

