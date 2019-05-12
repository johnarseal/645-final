import numpy as np
import scipy.ndimage
import skimage.color
import scipy.spatial.distance
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def singleConvolve(imgPath):
    '''
    Extracts the filter responses for the given image.

    [input]
    * imgPath: path of a single image
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

     # Read the image
    image = skimage.io.imread(imgPath)
    image = image.astype('float')/255
    
    if len(image.shape) == 2:
        image = np.tile(image[:,:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]
    
    image = skimage.color.rgb2lab(image)

    scales = [1,2,4,8,8*np.sqrt(2)]
    imgs = np.empty((image.shape[0], image.shape[1], 3*len(scales)*4))
    rt_ind = 0
    for i in range(5):
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            imgs[:,:,rt_ind] = img
            rt_ind += 1
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs[:,:,rt_ind] = img
            rt_ind += 1
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs[:,:,rt_ind] = img
            rt_ind += 1
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs[:,:,rt_ind] = img
            rt_ind += 1 
    
    return imgs


def display_filter_responses(response_maps, name="test"):
    '''
    Visualizes the filter response maps.

    [input]
    * response_maps: a numpy.ndarray of shape (H,W,3F)
    '''
    
    fig = plt.figure(1)
    
    for i in range(20):
        plt.subplot(5,4,i+1)
        resp = response_maps[:,:,i*3:i*3+3]
        resp_min = resp.min(axis=(0,1),keepdims=True)
        resp_max = resp.max(axis=(0,1),keepdims=True)
        resp = (resp-resp_min)/(resp_max-resp_min)
        plt.imshow(resp)
        plt.axis("off")

    plt.savefig(name +".jpg")
    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
    plt.show()