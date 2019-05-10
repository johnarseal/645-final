/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

/*
static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}*/

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ 
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
    unsigned short *membershipChanged = (unsigned short *)sharedMemory;
    float *clusters = (float *)(sharedMemory + blockDim.x * sizeof(unsigned short));

    membershipChanged[threadIdx.x] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}


// blockDim must be PowerOfTwo
__global__
void compute_delta(int *srcIntermediates,   // The source to fetch the intermediates
                   int *dstIntermediates,   // The destination to store the sum
                   int numIntermediates)    //  The actual number of intermediates
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (tId < numIntermediates) ? srcIntermediates[tId] : 0;

    __syncthreads();

    //  blockDim.x *must* be a power of two!
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dstIntermediates[blockIdx.x] = intermediates[0];
    }
}


// blockDim must be PowerOfTwo
__global__
void update_centroids_clusterSize(float *objects,           //  [numCoords][numObjs]
                    int *membership,          //  [numObjs]
                   float *centroids,    //  [numBlks][numCoords][numClusters]
                   int *clusterSize, // [numBlks][numClusters]
                   int numCoords, int numObjs, int numClusters)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if(tId < numObjs){
        int index = membership[tId];
        // update cluster size
        atomicAdd(&clusterSize[blockIdx.x*numClusters + index],1);
        for(int j = 0; j < numCoords; j++){
            int data_index = blockIdx.x*numCoords*numClusters+j*numClusters+index;
            int object_index = j*numObjs+tId;
            atomicAdd(&centroids[data_index],objects[object_index]);
        }
    }
}

/*
To be launched in 1 block, block size of numClusters
*/
__global__
void reduce_clusterSize(int *srcClusterSize, //  [numSrcBlks][numClusters]
                   int *dstClusterSize, // [numClusters]
                   int numSrcBlks, int numClusters){
    int tx = threadIdx.x;
    for(int i=0; i < numSrcBlks; i++){
        dstClusterSize[tx] += srcClusterSize[i*numClusters+tx];
    }     
}

/*
To be launched in numCoords block, block size of numClusters
*/
__global__
void reduce_centroids(float *srcCentroids,  //  [numSrcBlks][numCoords][numClusters]
                    float *dstCentroids,          //  [numCoords][numClusters]
                    int *clusterSize,       //     [numClusters]
                    int numSrcBlks, int numClusters, int numCoords)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    for(int i = 0; i < numSrcBlks; i++){
        dstCentroids[bx*numClusters+tx] += srcCentroids[i*numCoords*numClusters+bx*numClusters+tx];
    }
    dstCentroids[bx*numClusters+tx] /= clusterSize[tx];
}


