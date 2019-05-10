import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os, math, sys, getopt

# a helper function 
def nextPowerOfTwo(x):
    return 2**(x-1).bit_length()


def getHostDevicePair(shape, dtype_, initValue=1):
    hostMem = np.ones(shape,dtype=dtype_) * initValue
    deviceMem = cuda.mem_alloc(hostMem.nbytes)
    cuda.memcpy_htod(deviceMem, hostMem)
    return hostMem, deviceMem

    
# @TODO, this could be improved  
def init_centroids(objects, numClusters):
    return objects[:,:numClusters].copy()
    
    
    
def kmeans(objects, numClusters, threshold):
    """
    objects: numCoords x numObjs
    """
    
    event = cuda.Event()
    
    """ Step 0 cast to float, copy to device """
    objects = objects.astype(np.float32)
    objects_gpu = cuda.mem_alloc(objects.nbytes)
    cuda.memcpy_htod(objects_gpu, objects)
    numCoords, numObjs = objects.shape
    
    """ Step 1. Load cuda module """
    src = open("cuda_kmeans.cu").read()
    mod = SourceModule(src, include_dirs=[os.getcwd()])
    find_nearest_cluster = mod.get_function("find_nearest_cluster")
    compute_delta = mod.get_function("compute_delta")
    reduce_clusterSize = mod.get_function("reduce_clusterSize")
    reduce_centroids = mod.get_function("reduce_centroids")
    update_centroids_clusterSize = mod.get_function("update_centroids_clusterSize")
    
    """ Step 2. define some constant """
    # For find_nearest_cluster
    threadsPer_FNC_Block = 128
    num_FNC_Blocks = int(math.ceil(float(numObjs) / threadsPer_FNC_Block))
    # SDSize = shared memory size
    FNC_SDSize = threadsPer_FNC_Block * 2 + numClusters * numCoords * 4;
    # For compute_delta
    threadsPer_CD_Block = 128 if num_FNC_Blocks > 128 else nextPowerOfTwo(num_FNC_Blocks)
    num_CD_Blocks = int(math.ceil(float(num_FNC_Blocks) / threadsPer_CD_Block))
    CD_SDSize = threadsPer_CD_Block * 4
    
    """ Step 3. Init centroids using first K elements, define some variables """
    centroids =  init_centroids(objects, numClusters)
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    cuda.memcpy_htod(centroids_gpu, centroids)
    
    _,interm_gpu = getHostDevicePair((num_FNC_Blocks,),np.int32,0) # interm means intermediate
    membership, membership_gpu = getHostDevicePair((numObjs,),np.int32,-1)  # initialize membership to -1     
    reduceInterm, reduceInterm_gpu = getHostDevicePair((num_CD_Blocks,),np.int32,0)
    clusterSize, clusterSize_gpu = getHostDevicePair((numClusters,),np.int32,0)
    # seg means segregated
    segClusterSize, segClusterSize_gpu = getHostDevicePair((num_FNC_Blocks,numClusters),np.int32,0)
    _, segCentroids_gpu = getHostDevicePair((num_FNC_Blocks,numCoords,numClusters),np.int32,0)
    
    for loop in range(500):
        find_nearest_cluster(np.int32(numCoords), np.int32(numObjs), np.int32(numClusters), 
                objects_gpu, centroids_gpu, membership_gpu, interm_gpu,
                block=(threadsPer_FNC_Block,1,1),grid=(num_FNC_Blocks,1),shared=FNC_SDSize)
        event.synchronize()        
        
        
        """validating centroids"""
        """
        cuda.memcpy_dtoh(membership, membership_gpu)   
        cent_valid = np.zeros_like(centroids)
        clusterSize_valid = np.zeros_like(clusterSize)
        for i in range(numObjs):
            clusterSize_valid[membership[i]] += 1
            cent_valid[:,membership[i]] += objects[:,i] 
        
        cent_valid = cent_valid / clusterSize_valid
        print("\nvalid")
        print(cent_valid)
        """
        
        compute_delta(interm_gpu, reduceInterm_gpu, np.int32(num_FNC_Blocks),
                      block=(threadsPer_CD_Block,1,1),grid=(num_CD_Blocks,1), shared=CD_SDSize)
        event.synchronize()
        
        cuda.memcpy_dtoh(reduceInterm, reduceInterm_gpu)    
        event.synchronize()
        delta = reduceInterm.sum()

        # set segClusterSize and segCentroids to 0
        cuda.memset_d32(clusterSize_gpu, 0, numClusters)
        cuda.memset_d32(segClusterSize_gpu, 0, num_FNC_Blocks*numClusters)
        cuda.memset_d32(centroids_gpu, 0, numCoords*numClusters)
        cuda.memset_d32(segCentroids_gpu, 0, num_FNC_Blocks*numCoords*numClusters)
        
        event.synchronize()
        
        update_centroids_clusterSize(objects_gpu, membership_gpu, segCentroids_gpu, segClusterSize_gpu,
                                np.int32(numCoords), np.int32(numObjs), np.int32(numClusters), 
                                block=(threadsPer_FNC_Block,1,1), grid=(num_FNC_Blocks,1))
        event.synchronize()
        
        
        reduce_clusterSize(segClusterSize_gpu, clusterSize_gpu, np.int32(num_FNC_Blocks), np.int32(numClusters),
                        block=(numClusters,1,1))
        event.synchronize()
        
        reduce_centroids(segCentroids_gpu, centroids_gpu, clusterSize_gpu,
                        np.int32(num_FNC_Blocks), np.int32(numClusters), np.int32(numCoords),
                        block=(numClusters,1,1), grid=(numCoords,1))
        event.synchronize()
        
        """
        cuda.memcpy_dtoh(centroids, centroids_gpu)    
        print("computed centroids")
        print(centroids)         
        """
        
        delta /= float(numObjs)
        if delta <= threshold:
            break
    
    loop += 1
    cuda.memcpy_dtoh(centroids, centroids_gpu)    
    #print(centroids)
    
    print("Looped for", loop, "iterations")
    return centroids
        
        
def test_compute_delta():
    src = open("cuda_kmeans.cu").read()
    mod = SourceModule(src, include_dirs=[os.getcwd()])
    compute_delta = mod.get_function("compute_delta")    
    numInterm = 1e7
    numThreadsPerClusterBlock = 1024
    num_FNC_Blocks = int(math.ceil(float(numInterm) / numThreadsPerClusterBlock))
    
    totalInterms = np.ones((numInterm,),dtype=np.int32)
    
    interms = np.zeros((num_FNC_Blocks,), dtype=np.int32)
    
    compute_delta = mod.get_function("compute_delta")
    print("shared memory size",numThreadsPerClusterBlock*4)
    compute_delta(cuda.In(totalInterms),cuda.InOut(interms),np.int32(numInterm), block=(numThreadsPerClusterBlock,1,1), 
                    grid=(num_FNC_Blocks,1), shared=numThreadsPerClusterBlock*4)
    print(interms.sum())

    
def test_kmeans(path, K):
    lines = open(path).read().splitlines()
    numCoords = len(lines[0].split())-1
    objects = np.empty((numCoords,len(lines)))
    
    for i, line in enumerate(lines):
        nums = line.split()[1:]
        for j, n in enumerate(nums):
            objects[j,i] = float(n)

    print("finished parsing data")
    kmeans(objects, K, 0.001)
     

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"n:i:")
    for opt, arg in opts:
        if opt == "-n":
            numK = int(arg)
        if opt == "-i":
            path = arg
    test_kmeans(path, numK)
    

