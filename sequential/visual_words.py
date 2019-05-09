import numpy as np
from multiprocessing import Process, Value, Array
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import ctypes as c
import config

def f(rt_img, image, scale, ind, img_shape):
    image = np.frombuffer(image.get_obj()).reshape(img_shape)
    rt_img = np.frombuffer(rt_img.get_obj()).reshape((*img_shape[:2],60))

    rt_ind = ind*12
    for c in range(3):
        img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scale)
        rt_img[:,:,rt_ind] = img
        rt_ind += 1
    for c in range(3):
        img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scale)
        rt_img[:,:,rt_ind] = img
        rt_ind += 1
    for c in range(3):
        img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scale,order=[0,1])
        rt_img[:,:,rt_ind] = img
        rt_ind += 1
    for c in range(3):
        img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scale,order=[1,0])
        rt_img[:,:,rt_ind] = img
        #print(rt_img[:,:,rt_ind])
        rt_ind += 1   

        
def extract_filter_responses_mp(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    if len(image.shape) == 2:
        image = np.tile(image[:,:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]
    
    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    rt_img = np.empty((*image.shape[:2], 60))
    rt_img_share = Array(c.c_double, rt_img.flatten())
    image_share = Array(c.c_double, image.flatten())
    pQue = []
    for i in range(len(scales)):
        p = Process(target=f, args=(rt_img_share, image_share, scales[i], i, image.shape))
        p.start()
        pQue.append(p)
    
    for p in pQue:
        p.join()
    
    
    rt_img_share = np.frombuffer(rt_img_share.get_obj()).reshape((*image.shape[:2],60))
    
    return rt_img_share


def extract_filter_responses_rv(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    if len(image.shape) == 2:
        image = np.tile(image[:,:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]
    
    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    imgs = np.empty((*image.shape[:2], 3*len(scales)*4))
    rt_ind = 0
    for i in range(len(scales)):
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
    

def extract_filter_responses_old(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    if len(image.shape) == 2:
        image = np.tile(image[:,:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]
    
    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
    
    return imgs

    
def extract_filter_responses(image):
    return extract_filter_responses_rv(image)
    
    
def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    * dictionary: numpy.ndarray of shape (K, 60) where K is the number of cluster
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    response = extract_filter_responses(image).reshape((-1,60))
    distMat = scipy.spatial.distance.cdist(response, dictionary)
    wordmap = np.argmin(distMat, axis=1).reshape((*image.shape[:2]))
    
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * alpha: number of random samples
    * image_path: path of an image file
    * path_pref: saving path prefix

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    alpha,image_path,path_pref = args
    
    
    img = np.array(Image.open(image_path))
    filter_response = extract_filter_responses(img).reshape((-1,60))
    np.random.shuffle(filter_response)
    sample_response = filter_response[:alpha,:]
    np.save(path_pref+os.path.basename(image_path).split(".")[0],sample_response)
    
    
def compute_dictionary_sub_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * alpha: number of random samples
    * image_paths: path of multiple image file
    * path_pref: saving path prefix

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    alpha,image_paths,path_pref = args
    
    for img_path in image_paths:
        img = np.array(Image.open(img_path))
        filter_response = extract_filter_responses(img).reshape((-1,60))
        np.random.shuffle(filter_response)
        sample_response = filter_response[:alpha,:]
        np.save(path_pref+os.path.basename(img_path).split(".")[0],sample_response)    
    

def compute_dictionary(train_data=None, num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    * train_data: the loaded npz file that contains data["files"] as array of file name
                    each file name must be prepended by "../data/"
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    
    # ----- Set the Parameters HERE ----- #
    numCent = config.DICT_CENT
    alpha = config.SAMPLE_ALPHA
    save_path = "../data/response_alpha_"+str(alpha)+"/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    # if passing the train_data argument, Re-compute the image response
    if train_data is not None:    
        numD = len(train_data["files"])
        pathsArr = [[] for i in range(num_workers)]
        for i, path in enumerate(train_data["files"]):
            pathsArr[i%num_workers].append("../data/"+path)
        
        pQue = []
        for i in range(num_workers):
            p = Process(target=compute_dictionary_sub_image, 
                args=((alpha,pathsArr[i]),save_path,))
            pQue.append(p)
            p.start()
        
        for p in pQue:
            p.join()
        print("Finished filtering, loading responses")
    else:
        print("Already have filter response, loading responses")
    
    fns = os.listdir(save_path)
    numF = len(fns)
    responses = np.empty((alpha*numF, 60))
    for i,fp in enumerate(fns):
        responses[i*alpha:(i+1)*alpha,:] = np.load(save_path+fp)
    
    print("Finish loading nparray, start doing kmeans")
    kmeans = sklearn.cluster.KMeans(n_clusters=numCent).fit(responses) 
    dictionary = kmeans.cluster_centers_
    
    print("Complete K-means, logging centroids")
    print(dictionary)
    np.save(config.DICT_PATH,dictionary)
        
        
        
