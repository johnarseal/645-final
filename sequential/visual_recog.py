import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import scipy
import config
from multiprocessing import Process, Value, Array
from PIL import Image
from sklearn.metrics import confusion_matrix

def dump_features(ind, paths, labels, prefix):
    
    dictionary = np.load(config.DICT_PATH)
    features = np.empty((len(paths), config.NUM_F),dtype=float)
    for i,path in enumerate(paths):
        features[i] = get_image_feature("../data/"+path,dictionary,config.NUM_LAYER,config.DICT_CENT)
        if i % 100 == 0:
            print("process",ind,"got feature of",i,"images")
            
    np.savez(config.SYS_DIR+prefix+str(ind)+".npz", features=features,labels=labels)
   
    
def build_recognition_system():
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    
    print("Building system for NUM_LAYER:",config.NUM_LAYER,",cluster center:",
                    config.DICT_CENT,"SAMPLE ALPHA:",config.SAMPLE_ALPHA)
    print("Using dictionary",config.DICT_PATH)
    
    if not os.path.isdir(config.SYS_DIR):
        os.makedirs(config.SYS_DIR)
    
    print("Result will be saved into",config.SYS_DIR)
    
    
    train_data = np.load("../data/train_data.npz")
    prefix = "trained_system_"
    numTrain = len(train_data["files"])
    splitInd = [i*(numTrain//num_workers) for i in range(1,num_workers)]
    
    pathArr = np.split(train_data["files"], splitInd)
    labelArr = np.split(train_data["labels"], splitInd)
    
    pQue = []
    for i in range(num_workers):
        p = Process(target=dump_features, 
                args=(i, pathArr[i], labelArr[i], prefix))
        pQue.append(p)
        p.start()

    for p in pQue:
        p.join()

    print("all sub-process terminated, now merging the result")
    for i in range(num_workers):
        sub_system = np.load(config.SYS_DIR+prefix+str(i)+".npz")
        if i == 0:
            features = sub_system["features"]
        else:
            features = np.concatenate((features,sub_system["features"]),axis=0)
            
    np.savez(config.SYS_DIR+"trained_system.npz", dictionary=np.load(config.DICT_PATH),
            features=features,labels=train_data["labels"],SPM_layer_num=config.NUM_LAYER)
        
        
        
def evaluate_recognition_system(num_workers=4):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    
    test_data = np.load("../data/test_data.npz")
    
    trained_system = np.load(config.SYS_DIR+"trained_system.npz")
    prefix = "test_features"
    
    """
    #1. dump the test features
    numTest = len(test_data["files"])
    splitInd = [i*(numTest//num_workers) for i in range(1,num_workers)]
    
    pathArr = np.split(test_data["files"], splitInd)
    labelArr = np.split(test_data["labels"], splitInd)
    
    pQue = []
    for i in range(num_workers):
        p = Process(target=dump_features, 
                args=(i, pathArr[i], labelArr[i], prefix))
        pQue.append(p)
        p.start()

    for p in pQue:
        p.join()
    """
    
    print("all sub-process terminated, now merging the result")
    for i in range(num_workers):
        sub_system = np.load(config.SYS_DIR+prefix+str(i)+".npz")
        if i == 0:
            features = sub_system["features"]
        else:
            features = np.concatenate((features,sub_system["features"]),axis=0)
    
    print("Now evaluating the result")
    distMat = scipy.spatial.distance.cdist(features, trained_system["features"], 
            metric=lambda u, v: np.minimum(u,v).sum())
    
    testPredictInd = np.argmax(distMat,axis=1)
    
    testPredict = trained_system["labels"][testPredictInd]
    
    
    for i, label in enumerate(testPredict):
        if label != test_data["labels"][i]:
            print(test_data["files"][i], label)
    
    
    confMat = confusion_matrix(test_data["labels"],testPredict,labels=range(8))
    acc = confMat.trace()/confMat.sum()
    
    print(confMat)
    print("Accuracy",acc)
    
    return confMat, acc    



def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: feature (might be SPM) of an image
    '''
    
    image = np.array(Image.open(file_path))
    wordmap = visual_words.get_visual_words(image,dictionary)
    return get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    
    

def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    
    return scipy.spatial.distance.cdist(word_hist, histograms, metric=lambda u, v: np.minimum(u,v).sum()).flatten()




def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # the index of wordmap starts from 0
    hist = np.histogram(wordmap,bins=range(dict_size+1))
    return hist[0] / hist[0].sum()   



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    # ----- TODO -----
    H,W = wordmap.shape
    
    
    # the final vector we are going to return
    final_vec = np.empty((int(dict_size*(4**(layer_num+1)-1)/3),),dtype=float)
    curScale = 0.5
    final_ind = 0
    
    # compute the largest layer first
    num_cell = 2 ** layer_num
    bH, bW = H//num_cell, W//num_cell
    finestHistMap = np.empty((num_cell, num_cell, dict_size))
    for i in range(num_cell):
        for j in range(num_cell):
            sH, sW = i*bH, j*bW
            eH, eW = (i+1)*bH if i < num_cell-1 else H, (j+1)*bW if j < num_cell-1 else W
            finestHistMap[i,j,:] = get_feature_from_wordmap(wordmap[sH:eH,sW:eW],dict_size)
            final_vec[final_ind*dict_size:(final_ind+1)*dict_size] = finestHistMap[i,j] * curScale
            final_ind += 1
    
    prevHistMap = finestHistMap
    for l in range(layer_num-1,-1,-1):
        if l > 0:
            curScale = curScale / 2
        num_cell = num_cell//2
        curHistMap = np.empty((num_cell, num_cell, dict_size))
        for i in range(num_cell):
            for j in range(num_cell):
                curHistMap[i,j] = (prevHistMap[i*2,j*2] + prevHistMap[i*2+1,j*2] +
                                    prevHistMap[i*2,j*2+1] + prevHistMap[i*2+1,j*2+1])
                curHistMap[i,j] = curHistMap[i,j] / curHistMap[i,j].sum()
                final_vec[final_ind*dict_size:(final_ind+1)*dict_size] = curHistMap[i,j] * curScale
                final_ind += 1
                
        prevHistMap = curHistMap
        
    hist_all = final_vec / final_vec.sum()
    return hist_all
        
        
    
    






    

