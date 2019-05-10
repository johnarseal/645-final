import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os, math, time


def getHostDevicePair(shape, dtype_, initValue=1):
    hostMem = np.ones(shape,dtype=dtype_) * initValue
    deviceMem = cuda.mem_alloc(hostMem.nbytes)
    cuda.memcpy_htod(deviceMem, hostMem)
    return hostMem, deviceMem


def computeHistFeature(wordMap, numFeat, mod):
    """
    wordMap: numCoords(60) x numClusters
    imgConvolve: M x N x numCoords
    """
    
    """
    1. Define the variables needed
    """
    numPixel = wordMap.size
    numThreadsPerBlock = 128
    numBlocks = int(math.ceil(numPixel/float(numThreadsPerBlock)))
    _, segHistFeature_gpu = getHostDevicePair((numBlocks, numFeat), np.int32, initValue=0)
    histFeature = np.zeros((numFeat, ), dtype=np.int32)
    
    """
    2. run the function
    """    
    count_hist_feat = mod.get_function("count_hist_feat")
    reduce_hist_feat = mod.get_function("reduce_hist_feat")
    count_hist_feat(cuda.In(wordMap), segHistFeature_gpu, np.int32(numFeat), np.int32(numPixel),
                    block=(numThreadsPerBlock,1,1),grid=(numBlocks,1))
    reduce_hist_feat(segHistFeature_gpu, cuda.InOut(histFeature), np.int32(numFeat), np.int32(numBlocks),
                    block=(numFeat,1,1))
    """
    3. normalize and return
    """
    histFeature = histFeature.astype(np.float32) / numPixel
    return histFeature


def getHistFeature(centroids,imgConvolve, layer_num, mod):
    """
    centroids: numCoords(60) x numClusters
    imgConvolve: M x N x numCoords
    """
    
    """
    1. get function
    """
    find_nearest_cluster = mod.get_function("find_nearest_cluster")

    """
    2. reshape, transpose and cast imgConvolve
    """
    numCoords, numClusters = centroids.shape
    imH, imW, _ = imgConvolve.shape
    imgConvolve = imgConvolve.reshape((-1, numCoords)).T # [numCoords x numPixel]
    imgConvolve = imgConvolve.astype(np.float32)  
    numPixel = imgConvolve.shape[1]
    
    """
    3. Define needed variables
    """
    wordMap = np.empty((numPixel,),dtype=np.int32)
    numThreadsPerBlock = 1024
    numBlocks = int(math.ceil(numPixel/float(numThreadsPerBlock)))
    
    """
    4. find_nearest_cluster to get wordMap
    """
    find_nearest_cluster(np.int32(numCoords), np.int32(numPixel), np.int32(numClusters),
                         cuda.In(imgConvolve), cuda.In(centroids), cuda.InOut(wordMap),
                         block=(numThreadsPerBlock,1,1), grid=(numBlocks,1), shared=centroids.size*4)
    
    
    """
    Layer by layer construct the histogram features
    """
    final_vec = np.empty((int(numClusters*(4**(layer_num+1)-1)/3),),dtype=np.float32)
    curScale = 0.5
    final_ind = 0
    
    """
    compute the largest layer first
    """
    num_cell = 2 ** layer_num           # num cell along the border
    bH, bW = imH//num_cell, imW//num_cell   # the height and width of each cell
    finestHistMap = np.empty((num_cell, num_cell, numClusters), np.float32)
    for i in range(num_cell):
        for j in range(num_cell):
            sH, sW = i*bH, j*bW
            eH, eW = (i+1)*bH if i < num_cell-1 else H, (j+1)*bW if j < num_cell-1 else W
            finestHistMap[i,j,:] = computeHistFeature(wordMap[sH:eH,sW:eW], numClusters, mod)
            final_vec[final_ind*numClusters:(final_ind+1)*numClusters] = finestHistMap[i,j] * curScale
            final_ind += 1
    
    """
    construct the lower-layered histogram by aggregating the finer histogram
    There is currently no parallel on the below work
    """
    prevHistMap = finestHistMap
    for l in range(layer_num-1,-1,-1):
        if l > 0:
            curScale = curScale / 2
        num_cell = num_cell//2
        curHistMap = np.empty((num_cell, num_cell, numClusters))
        for i in range(num_cell):
            for j in range(num_cell):
                curHistMap[i,j] = (prevHistMap[i*2,j*2] + prevHistMap[i*2+1,j*2] +
                                    prevHistMap[i*2,j*2+1] + prevHistMap[i*2+1,j*2+1])
                curHistMap[i,j] = curHistMap[i,j] / curHistMap[i,j].sum()
                final_vec[final_ind*numClusters:(final_ind+1)*numClusters] = curHistMap[i,j] * curScale
                final_ind += 1
                
        prevHistMap = curHistMap
        
    hist_all = final_vec / final_vec.sum()
    return hist_all

def test_computeHistFeature():
    numFeat = 128
    wordMap = np.random.randint(0,numFeat,size=(520,520)).astype(np.int32)
    t1 = time.time()
    hist_valid = np.histogram(wordMap,bins=range(numFeat+1))[0].astype(np.float32)
    hist_valid = hist_valid / (520*520)
    t2 = time.time()
    print("sequential time:",t2-t1)
    print(hist_valid)
    print(hist_valid.sum())
    
    src = open("cuda_hist_feat.cu").read()
    mod = SourceModule(src)
    
    start = cuda.Event()
    stop = cuda.Event()
    cuda.Context.synchronize()
    start.record()
    
    hist = computeHistFeature(wordMap.flatten(), numFeat, mod)
    
    stop.record()
    stop.synchronize()
    elapsed_seconds = stop.time_since(start)*1e-3
    print("cuda time:", elapsed_seconds)

    
    print(hist)
    print(hist.sum())

if __name__ == '__main__':
    test_computeHistFeature()
