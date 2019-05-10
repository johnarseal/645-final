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
// This is a simplfied version of the same-name function in cuda_kmeans.cuda_kmeans
// we just want to find the nearest cluster, and don't care about the membership change
__global__ 
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership)          //  [numObjs]
{
    extern __shared__ char sharedMemory[];
    float *clusters = (float *)sharedMemory;

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

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]
    }
}


// inside each block count the histogram
__global__ 
void count_hist_feat(int *originData,          // [numOriginData]
                     int *segHist,             // [numBlks][numFeatures]
                     int numFeatures,
                     int numOriginData)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if(tId < numOriginData){
        int featInd = originData[tId];
        //printf("In thread %d, featInd: %d \n", tId, featInd);
        atomicAdd(&segHist[blockIdx.x*numFeatures + featInd],1);
    }
}


// aggregate segregated histogram
// to be launched in 1 block of (numFeatures,)
__global__ 
void reduce_hist_feat(int *segHist,              // [numBlks][numFeatures]
                     int *finalHist,             // [numFeatures]
                     int numFeatures,
                     int numBlks)               // numBlocks
{
    for(int i = 0; i < numBlks; i++){
        //printf("In thread %d, value: %d \n", threadIdx.x, segHist[i*numFeatures+threadIdx.x]);
        finalHist[threadIdx.x] += segHist[i*numFeatures+threadIdx.x];
    }
}




