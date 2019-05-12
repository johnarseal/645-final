import numpy as np
import skimage.io
import scipy.ndimage
import os
import fnmatch

import sequential
import cuda_convolve as cc

import timeit

def batchSampleConvolve(directory, alpha=200):
    """
    directory: directory of the images
    alpha: a constant, usually 200  
    return mat: (K * alpha) * 60 matrix. Where K is the number of images
    """

    imgPathList = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            imgPathList.append(os.path.join(root, filename))

    result = np.empty((len(imgPathList) * alpha, 60))

    # Perfrom convolvution on each image
    for i, path in enumerate(imgPathList):
        imgs = singleConvolve(path)
        # sequential.display_filter_responses(imgs, str(i))

        # Sample vectors
        for j in range(alpha):
            rndIdx1 = np.random.randint(0, imgs.shape[0])
            rndIdx2 = np.random.randint(0, imgs.shape[1])
            result[i*alpha+j] = imgs[rndIdx1][rndIdx2]
    
    print(result.shape)
    return result

def singleConvolve(imgPath):
    """
    imgPath: path of a single image
    return mat: M x N x 60 matrix. Where M and N is the shape of the image
    """

    # Read the image
    image = skimage.io.imread(imgPath)
    image = image.astype('float')/255
    
    # Conver grey-scale and RGBA to RGB
    if len(image.shape) == 2:
        image = np.tile(image[:,:, np.newaxis], (1, 1, 3))
    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)

    scales = [1,2,4,8,8*np.sqrt(2)]
    
    imgs = np.empty(shape=(image.shape[0],image.shape[1], len(scales)*3*4))
    
    rt_ind = 0
    for i in range(len(scales)):

        # Generate the filters
        gaussian_filter = cc.gaussian_kernel(sigma=scales[i])
        derivative_of_gaussian_filter = cc.derivative_of_gaussian_kernel(sigma=scales[i])

        # Gaussian
        for c in range(3):
            original = np.float32(image[:,:,c])
            destImage = original.copy()
            destImage[:] = np.nan
            destImage = cc.convolution_cuda(original,  gaussian_filter,  gaussian_filter)
            imgs[:,:,rt_ind] = destImage
            rt_ind += 1
    
        # Laplace
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs[:,:,rt_ind] = img
            rt_ind += 1
    
        for c in range(3):
            original = np.float32(image[:,:,c])
            destImage = original.copy()
            destImage[:] = np.nan
            destImage = cc.convolution_cuda(original,  derivative_of_gaussian_filter,  derivative_of_gaussian_filter)
            imgs[:,:,rt_ind] = destImage
            rt_ind += 1

        for c in range(3):
            original = np.float32(image[:,:,c])
            destImage = original.copy()
            destImage[:] = np.nan
            destImage = cc.convolution_cuda(original,  gaussian_filter,  derivative_of_gaussian_filter)
            imgs[:,:,rt_ind] = destImage
            rt_ind += 1
    

    return imgs


def testSingleConvolve():
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    imgs = singleConvolve(path_img)
    sequential.display_filter_responses(imgs)

def testBatchSampleConvolve():
    start = timeit.default_timer()
    batchSampleConvolve("../data")
    stop = timeit.default_timer()
    print('Runtime: ', stop - start)  
    

if __name__ == '__main__':
    # testSingleConvolve()
    testBatchSampleConvolve()
