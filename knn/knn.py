
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os, math, sys, getopt
import matplotlib.pyplot as plt
K = 1
threads_per_block = 16
    
def knn(traning_data, test_data): 
    '''
     training_data: dim(400) * train_num(1500) 
     test_data: dim(400) *  test_num
     train_label: 1 * 1500
    '''
    event = cuda.Event()
    
    """ Step 0 prepare data and allocate memory """
    training_data = training_data.astype(np.float32) # ref_dev
    traning_data_gpu = cuda.mem_alloc(traning_data.nbytes)
    cuda.memcpy_htod(traning_data_gpu, traning_data)
    train_dim, train_num = training_data.shape
    print("train_dim :"+train_dim + " train_num: "+train_num)

    test_data = test_data.astype(np.float32)  # query_dev
    test_data_gpu = cuda.mem_alloc(test_data.nbytes)
    cuda.memcpy_htod(test_data_gpu, test_data)
    test_dim, test_num = test_data.shape
    print("test_dim :"+test_dim + " test_num: "+test_num)

    similarity = np.zeros([test_num, train_num])
    similarity = similarity.astype(np.float32)
    similarity_gpu = cuda.mem_alloc(similarity.nbytes)
    cuda.memcpy_htod(similarity_gpu, similarity)

    index = np.zeros([test_num, k])
    index = index.astype(np.float32)
    index_gpu = cuda.mem_alloc(index.nbytes)
    cuda.memcpy_htod(index_gpu, index)
    
    """ Step 1. Load cuda module """
    src = open("knnCUDA.cu").read()
    mod = SourceModule(src, include_dirs=[os.getcwd()])
    compute_dist_global = mod.get_function("cuComputeDistanceGlobal")
    insertion_sort = mod.get_function("cuInsertionSort")
    
    """ Step 2. define some constant """
    dim_grid = (math.ceil(float(train_num) / threads_per_block), math.ceil(float(test_num) / threads_per_block), 1)
    dim_block = (threads_per_block, threads_per_block, 1)
    
    """ Step 3. Init centroids using first K elements, define some variables """
    compute_dist_global(training_data_gpu, np.int32(train_num), test_data_gpu, 
        np.int32(test_num), np.int32(test_dim), similarity_gpu, block = dim_block, grid = dim_grid)

    event.synchronize()

    insertion_sort(similarity_gpu, index_gpu, np.int32(test_num), 
        np.int32(train_num), K, block = dim_block, grid = dim_grid)

    event.synchronize()

    cuda.memcpy_dtoh(similarity, similarity_gpu)
    cuda.memcpy_dtoh(index, index_gpu)  
    print(similarity)
    print(index)

def generate_hist(dim, num):
    data = []
    for i in range(num):
        mu = 85
        sigma = 4
        np.random.seed(0)
        s = np.random.normal(mu, sigma, dim)
        plt.hist(s, 30, normed=True) 
        data.append(s)

def test_knn():
    training_data = generate_hist(400, 1500)
    test_data = generate_hist(400, 1)
    knn(training_data, test_data)

if __name__ == '__main__':
    # generate 
    knn(training_data, test_data)