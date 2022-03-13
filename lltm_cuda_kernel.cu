#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_runtime.h"

#include <cstdint>
#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <vector>
//#include <cuda/annotated_ptr>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

#include <cmath>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <assert.h>
#include "device_launch_parameters.h"
using namespace cooperative_groups;


#pragma once

//constants describing the meaning of main shared memory spaces
constexpr uint32_t localWorkQueLength = 300;
constexpr uint32_t startOfLocalWorkQ = 4160;
constexpr uint32_t lengthOfMainShmem = 4460;//4460;
constexpr uint32_t begResShmem = 1088;
constexpr uint32_t begfirstRegShmem = 2112;
constexpr uint32_t begSecRegShmem = 3136;
constexpr uint32_t begSMallRegShmemA = 0;
constexpr uint32_t begSMallRegShmemB = 1056;
constexpr uint32_t begSourceShmem = 32;

//as the pipeline is so asynchronous we need some additional storage for saving data related to the operations
//will define how big is the  amount of space in order to be able to use fast version of modulo operator it should be power of 2 ... 
constexpr uint32_t modForPipelineVars = 16;
constexpr uint32_t fpLocCountBeg = 4460;




// other
//added to linear index meta in order to  mark weather block is of type gold or not 
constexpr uint32_t  isGoldOffset = (UINT16_MAX * 10);


/**
In order to be able to use cuda malloc 3d we will implemnt it as a series
of 3d arrays
*/



#pragma once
template <typename TFPP>
struct array3dWithDimsCPU {
    //TFPP*** arrP;
    TFPP* arrP;
    int Nx;
    int Ny;
    int Nz;
};


#pragma once
template <typename TFPP>
struct array3dWithDimsGPU {
    TFPP* arrP;
    //cudaPitchedPtr arrPStr;
    //cudaPitchedPtr arrPStr;
    int Nx;
    int Ny;
    int Nz;
};


#pragma once
extern "C" struct MetaDataCPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;


    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count  9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter 
     //12) global FPandFn offset 13)globalIterationNumb
    //array3dWithDimsCPU<unsigned int> minMaxes;
    unsigned int* minMaxes;

    ////// counts of false positive and false negatives in given metadata blocks

    ///// sizes of array below will be established on the basis of fp and fn values known after boolKernel finished execution

    //work queue -  workqueue counter already present in minMaxes as entry 9 
    //in practice it is matrix of length the same as FP+FN global count +1 and width of 5
         //1) xMeta; 2)yMeta 3)zMeta 4)isGold 5)iteration number  
    //we use one single long rewsult list - in order to avoid overwriting each block each block has established offset where it would write it's results 
    uint32_t* resultList;

};

#pragma once
extern "C" struct MetaDataGPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;

    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count 9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
    //12) global FPandFn offset 13)globalIterationNumb

    unsigned int* minMaxes;

    //represents x from description of main Arr
    unsigned int mainArrXLength;
    //have length 4x
    unsigned int mainArrSectionLength;

    unsigned int metaDataSectionLength = 20;

    // now we will store here also calculated by min maxes kernel values of minimum and maximumvalues 
        //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ 
    unsigned int maxX;
    unsigned int minX;
    unsigned int maxY;
    unsigned int minY;
    unsigned int maxZ;
    unsigned int minZ;
};








/*
* Basically holding the arguments for master functions controlling full preparation to get all for Housedorff kernel
*/
#pragma once
template <typename TFF>
struct ForFullBoolPrepArgs {

    //metadata struct
    MetaDataCPU metaData;

    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    array3dWithDimsCPU<TFF> goldArr;
    array3dWithDimsCPU<TFF> segmArr;
    TFF numberToLookFor;// what we will look for in arrays
    //number and dimensionality of threads and blocks required to lounch bool kernel
    dim3 threads;
    int blocks;
    //threads and blocks for first metadata pass kernel
    int threadsFirstMetaDataPass;
    int blocksFirstMetaDataPass;
    //threads and blocks for main pass 
    dim3 threadsMainPass;
    int blocksMainPass;
    //threads and blocks for padding pass 
    dim3 threadsPaddingPass;
    int blocksPaddingPass;
    //threads and blocks for non first metadata passes 
    int threadsOtherMetaDataPasses;
    int blocksOtherMetaDataPasses;
    // will establish how many points we want to include in dilatation and how many we can ignore so typically set to 95% - so we will ignore only 5% most distant
    float robustnessPercent = 1.0;  // 0.95;

    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocalCPU;
    uint32_t* resultListPointerIterNumb;

};


/*
* Basically holding the arguments for main kernel in the FullBoolPrep
*/
#pragma once
template <typename TFB>
struct ForBoolKernelArgs {
    //matadata struct
    MetaDataGPU metaData;
    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    array3dWithDimsGPU<TFB> goldArr;
    array3dWithDimsGPU<TFB> segmArr;
    TFB numberToLookFor;


    // Frequent accesses to "a" and "b"; infrequent accesses to "x" and "y":
    //cuda::annotated_ptr<int const, cuda::access_property::persisting> a_p{ a }, b_p{ b };
    //cuda::annotated_ptr<int, cuda::access_property::streaming> x_s{ x }, y_s{ y };

    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;

    uint32_t* origArrsPointer;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;
    unsigned int* minMaxes;


    /*
main array with all required data  organized in sections for each metadata block
x-  is block dimx times block dimy
now what occupies what positions
##mainArrA
(0) - x-1 : reducedGoldRef
(x) - 2x-1 : reducedSegmRef
##mainArrB
() - 3x-1 : reducedGoldPrev
(x) - 4x-1 : reducedSegmPrev

##metaDataArr
0: empty
1 :fpCount
2 :fnCount
3 :fpCounter
4 :fnCounter
5 :fpOffset
6 :fnOffset
7 :isActiveGold
8 :isFullGold
9 :isActiveSegm
10 :isFullSegm
11 :isToBeActivatedGold
12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
13 : top
14 : bottom
15 : left
16 : right
17 : anterior
18 : posterior
19 : empty
20 : empty


###origArrs
0-x : reducedGold
(x+1) - 2x : reducedSegm

*/
//   uint32_t* mainArr;




    float robustnessPercent = 1.0;// 0.95;

};





//just utility for unit testing - set some data bout points
#pragma once
extern "C"  struct forTestPointStruct {
    int x;
    int y;
    int z;

    bool isGold;
    bool isGoldAndSegm;

    int xMeta;
    int yMeta;
    int zMeta;



    bool shouldBeInResAfterOneDil;
    bool shouldBeInResAfterTwoDil;
    bool isFoundAndDilatated = false;
    bool isFoundInResult = false;

    bool isFoundDilTop = false;
    bool isFoundDilBottom = false;

    bool isFoundDilAnterior = false;
    bool isFoundDilPosterior = false;

    bool isFoundDilLeft = false;
    bool isFoundDilRight = false;

};


#pragma once
extern "C" struct forTestMetaDataStruct {

    int xMeta;
    int yMeta;
    int zMeta;

    int requiredspaceInFpResultList;
    int requiredspaceInFnResultList;

    bool isToBeActiveAtStart;
    bool isToBeActiveAfterOneIter;
    bool isToBeActiveAfterTwoIter;

    bool isToBeValidatedFpAfterOneIter;
    bool isToBeValidatedFpAfterTwoIter;

    bool isToBeValidatedFnAfterOneIter;
    bool isToBeValidatedFnAfterTwoIter;


    bool isToBeFullAfterOneIter;
    bool isToBeFullAfterTwoIter;

    int fpCount;
    int fnCount;

    int fpConterAfterOneDil;
    int fpConterAfterTwoDil;

    int fnConterAfterOneDil;
    int fnConterAfterTwoDil;


};


/*
data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
*/
struct occupancyCalcData {
    int warpsNumbForMinMax;
    int blockSizeForMinMax;

    int warpsNumbForboolPrepareKernel;
    int blockSizeFoboolPrepareKernel;

    int theadsForFirstMetaPass;
    int blockForFirstMetaPass;

    int warpsNumbForMainPass;
    int blockForMainPass;
};



/*
copy from host to device
*/
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(MetaDataCPU metaDataCPU, unsigned int*& minMaxes) {
    MetaDataGPU res;

    metaDataCPU.minMaxes[1] = 0;
    metaDataCPU.minMaxes[2] = 1000;
    metaDataCPU.minMaxes[3] = 0;
    metaDataCPU.minMaxes[4] = 1000;
    metaDataCPU.minMaxes[5] = 0;
    metaDataCPU.minMaxes[6] = 1000;
    metaDataCPU.minMaxes[7] = 0;
    metaDataCPU.minMaxes[8] = 0;
    metaDataCPU.minMaxes[9] = 0;
    metaDataCPU.minMaxes[10] = 0;
    metaDataCPU.minMaxes[11] = 0;
    metaDataCPU.minMaxes[12] = 0;
    metaDataCPU.minMaxes[13] = 0;
    metaDataCPU.minMaxes[14] = 0;
    metaDataCPU.minMaxes[15] = 0;
    metaDataCPU.minMaxes[16] = 0;
    metaDataCPU.minMaxes[17] = 0;
    metaDataCPU.minMaxes[18] = 0;
    metaDataCPU.minMaxes[19] = 0;
    metaDataCPU.minMaxes[20] = 0;

    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxes, metaDataCPU.minMaxes, size, cudaMemcpyHostToDevice);

    //res.resultList = allocate3dInGPU(metaDataCPU.resultList);

    //res.metaXLength = metaDataCPU.metaXLength;
    //res.MetaYLength = metaDataCPU.MetaYLength;
    //res.MetaZLength = metaDataCPU.MetaZLength;

    //res.totalMetaLength = metaDataCPU.totalMetaLength;
    //allocating on GPU and copying  cpu data onto GPU

    return res;

}

/*
copy from device to host
*/
#pragma once
inline void copyMetaDataToCPU(MetaDataCPU metaDataCPU, MetaDataGPU metaDataGPU) {
    //copyDeviceToHost3d(metaDataGPU.fpCount, metaDataCPU.fpCount);
    //copyDeviceToHost3d(metaDataGPU.fnCount, metaDataCPU.fnCount);
    size_t size = sizeof(unsigned int) * 20;

    cudaMemcpy(metaDataCPU.minMaxes, metaDataGPU.minMaxes, size, cudaMemcpyDeviceToHost);
}


#pragma once
inline cudaError_t checkCuda(cudaError_t result, std::string description)
{
    if (result != cudaSuccess) {
        std::cout << description;
        std::cout << "\n";
        // printf("%d", description);
        fprintf(stderr, "CUDA Runtime Error in %d : %s\n", description, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}





/*
setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
*/
#pragma once
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs
    , uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd
    , uint32_t linIdexMeta, MetaDataGPU metaData, uint32_t localBlockMetaData[20]) {

    if ((threadIdx.x == idX) && (threadIdx.y == 0)) {
        if (predicate) {


            localBlockMetaData[inArrIndex] = (linIdexMeta + toAdd);
        }
        else {
            localBlockMetaData[inArrIndex] = isGoldOffset;
        }
    };
}

/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr, TYO* goldArr, TYO* segmArr, unsigned int* minMaxes) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;

    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;

    //auto pipeline = cuda::make_pipeline();


    //shared memory

    //TODO() make it dynamically sized 
    __shared__ uint32_t sharedForGold[1024];
    __shared__ uint32_t sharedForSegm[1024];


    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];
    __shared__ uint32_t localBlockMetaData[20];

    __shared__ bool anyInGold[1];
    __shared__ bool anyInSegm[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ int minMaxesInShmem[7];



    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) {
        anyInSegm[1] = false;

        if (blockIdx.x == 0) {
            printf("in bool kernel  dims meta in bool kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ \n "
                , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
                , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
                , metaData.minX, metaData.minY, metaData.minZ
            );
        }

    };



    sync(cta);

    /////////////////////////


    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int xMeta = int(linIdexMeta % (metaData.metaXLength));
        int zMeta = int(floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength))));
        int yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block
        sync(cta);
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nx) {

                    // resetting 
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nz) {
                            //char* tensorslice;

                            //first array gold
                            bool goldBool = goldArr[x + y * fbArgs.goldArr.Nx + z * fbArgs.goldArr.Nx * fbArgs.goldArr.Ny] == fbArgs.numberToLookFor;
                            bool segmBool = segmArr[x + y * fbArgs.segmArr.Nx + z * fbArgs.segmArr.Nx * fbArgs.segmArr.Ny] == fbArgs.numberToLookFor;
                            //goldBool = true;

                            // setting bits
                            sharedForGold[xLoc + yLoc * fbArgs.dbXLength] |= goldBool << zLoc;
                            sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;

                            //if (goldBool) {
                            //    printf("in kernel  gold x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z,  xMeta, yMeta,zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //    , linIdexMeta);
                            //}

                            //if (segmBool) {
                            //    printf("in kernel  segm  x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z,  xMeta, yMeta, zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //        , linIdexMeta);
                            //}


                        }
                    }
                }

                //if (sharedForGold[xLoc + yLoc * fbArgs.dbXLength] > 0) {
                //    printf("in kernel Metax %d yMeta %d zMeta %d linearLocal %d linIdexMeta %d column %d \n"
                //        , xMeta, yMeta, zMeta,  xLoc + yLoc * fbArgs.dbXLength, linIdexMeta
                //    , sharedForGold[xLoc + yLoc * fbArgs.dbXLength]);
                //}


            }
        }
        //reset local metadata
        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            localBlockMetaData[threadIdx.x] = 0;
        }



        isNotEmpty = __syncthreads_or(isNotEmpty);
        //exporting to global memory
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nx) {

                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

        //   sync(cta);



           //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]) , (sharedForGold), sizeof(uint32_t) * cta.size(), barrier);
           //barrier.arrive_and_wait(); // Waits for all copies to complete


          // cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          //barrier.arrive_and_wait(); // Waits for all copies to complete

          //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          //barrier.arrive_and_wait(); // Waits for all copies to complete

          //cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength ]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*1]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y) , barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*2]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*3]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

        sync(cta);



        /////adding the block and total number of the Fp's and Fn's 
        sumFp = reduce(tile, sumFp, plus<uint32_t>());
        sumFn = reduce(tile, sumFn, plus<uint32_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        sync(cta);//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();

        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    printf("xMeta %d yMeta %d zMeta %d \n", xMeta, yMeta, zMeta);
        //}

        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];
                /*               if (sharedForGold[i]>0) {
                                   printf("adding sharedForGold[i] %d in gold \n ", sharedForGold[i]);
                               }*/

            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            //metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 1] = sharedForGold[33];
            localBlockMetaData[1] = sharedForGold[33];

        }
        // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
        if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {


            sharedForSegm[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForSegm[33] += sharedForSegm[i];
            };
            fpSFnS[1] += sharedForSegm[33];// will be needed later for global set
            //setting metadata
            localBlockMetaData[2] = sharedForSegm[33];

        }

        //marking as active 
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) {
            localBlockMetaData[7] = 1;

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
            localBlockMetaData[9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved 
        //top



        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(metaData.metaXLength * metaData.MetaYLength)), linIdexMeta, metaData, localBlockMetaData);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (metaData.MetaZLength - 1)), (metaData.metaXLength * metaData.MetaYLength), linIdexMeta, metaData, localBlockMetaData);//bottom

        setNeighbourBlocks(fbArgs, 6, 15, (xMeta > 0), (-1), linIdexMeta, metaData, localBlockMetaData);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (metaData.metaXLength - 1)), 1, linIdexMeta, metaData, localBlockMetaData);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (metaData.MetaYLength - 1)), metaData.metaXLength, linIdexMeta, metaData, localBlockMetaData);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-metaData.metaXLength), linIdexMeta, metaData, localBlockMetaData);//posterior

        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + threadIdx.x] = localBlockMetaData[threadIdx.x];
        };

        sync(cta); // just to reduce the warp divergence

        // copy metadata to global memory

        //cuda::memcpy_async(cta, &metaDataArr[linIdexMeta * metaData.metaDataSectionLength], (&localBlockMetaData[0]), (sizeof(uint32_t) * 20), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

    }



    sync(cta);


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[8]), fpSFnS[1]);

    };
}



/*
    a) we define offsets in the result list to have the results organized and avoid overwiting
    b) if metadata block is active we add it in the work queue
*/


/*
we add here to appropriate queue data  about metadata of blocks of intrest
minMaxesPos- marks in minmaxes the postion of global offset counter -12) global FP offset 13) global FnOffset
offsetMetadataArr- arrays from metadata holding data about result list offsets it can be either fbArgs.metaData.fpOffset or fbArgs.metaData.fnOffset
*/


#pragma once
__device__ inline void addToQueue(uint32_t linIdexMeta, uint8_t isGold
    , unsigned int fpFnLocCounter[1], uint32_t localWorkQueue[1600], uint32_t localOffsetQueue[1600], unsigned int localWorkQueueCounter[1]
    , uint8_t countIndexNumb, uint8_t isActiveIndexNumb, uint8_t offsetIndexNumb
    , uint32_t* metaDataArr, MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue) {

    unsigned int count = metaDataArr[linIdexMeta * metaData.metaDataSectionLength + countIndexNumb];
    //given fp is non zero we need to  add this to local queue
    if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + isActiveIndexNumb] == 1) {

        // printf("adding to local in first meta pass linIdexMeta %d isGold %d isActiveIndexNumb %d \n  ", linIdexMeta, isGold, isActiveIndexNumb);

        count = atomicAdd_block(&fpFnLocCounter[0], count);
        unsigned int  old = atomicAdd_block(&localWorkQueueCounter[0], 1);
        //we check weather we still have space in shared memory
        if (old < 1590) {// so we still have space in shared memory
            // will be equal or above isGoldOffset  if it is gold pass
            localWorkQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            localOffsetQueue[old] = uint32_t(count);
        }
        else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
            old = atomicAdd(&(minMaxes[9]), old);
            //workQueue
            workQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            //and offset 
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + offsetIndexNumb] = atomicAdd(&(minMaxes[12]), count);
        };
    }
}


#pragma once
template <typename PYO>
__global__ void firstMetaPrepareKernel(ForBoolKernelArgs<PYO> fbArgs
    , MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* origArrs, uint32_t* metaDataArr) {

    //////initializations
    thread_block cta = this_thread_block();
    char* tensorslice;// needed for iterations over 3d arrays
   //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint32_t localWorkQueue[1600];
    __shared__ uint32_t localOffsetQueue[1600];
    if ((threadIdx.x == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 1)) {
        localWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 2)) {
        globalWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 3)) {
        globalOffsetForBlock[0] = 0;
    }
    sync(cta);


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {

        // if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
           //  printf("in first meta pass linIdexMeta %d blockIdx.x %d blockDim.x %d metaData.totalMetaLength %d threadIdx.x %d \n  ", linIdexMeta, blockIdx.x, blockDim.x, metaData.totalMetaLength, threadIdx.x );
         //}

         //goldpass
        addToQueue(linIdexMeta, 0
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 1, 9, 6
            , metaDataArr, metaData, minMaxes, workQueue);
        //segmPass  
        addToQueue(linIdexMeta, 1
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 2, 7, 5
            , metaDataArr, metaData, minMaxes, workQueue);



        /*       addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fpOffset, fbArgs.metaData.fpCount, 0, fbArgs.metaData.isActiveSegm, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
               addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fnOffset, fbArgs.metaData.fnCount, 1, fbArgs.metaData.isActiveGold, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);*/
    }
    sync(cta);
    if ((threadIdx.x == 0)) {
        globalOffsetForBlock[0] = atomicAdd(&(minMaxes[12]), (fpFnLocCounter[0]));

        /* if (fpFnLocCounter[0]>0) {
             printf("\n in meta first pass global offset %d  locCounter %d \n  ", globalOffsetForBlock[0], fpFnLocCounter[0]);
         }*/
    };
    if ((threadIdx.x == 1)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
        }
    }
    sync(cta);

    //exporting to global work queue
    //cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (localWorkQueue), (sizeof(uint32_t) * localWorkQueueCounter[0]));


    //setting offsets
    for (uint32_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        workQueue[globalWorkQueueCounter[0] + i] = localWorkQueue[i];

        /*        printf("FFIrst meta pass lin meta to Work Q %d is gold %d to spot %d  \n "
            , localWorkQueue[i] - isGoldOffset*(localWorkQueue[i] >= isGoldOffset)
                , (localWorkQueue[i] >= isGoldOffset), globalWorkQueueCounter[0] + i);*/

                //FP pass
        if (localWorkQueue[i] >= isGoldOffset) {
            metaDataArr[(localWorkQueue[i] - isGoldOffset) * metaData.metaDataSectionLength + 5] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fp offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i] - isGoldOffset
            //    , localOffsetQueue[i] + globalOffsetForBlock[0] 
            //, globalOffsetForBlock[0]
            //, localOffsetQueue[i]);

        }
        //FN pass
        else {
            metaDataArr[(localWorkQueue[i]) * metaData.metaDataSectionLength + 6] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fn offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i] 
            //    , localOffsetQueue[i] + globalOffsetForBlock[0]
            //    , globalOffsetForBlock[0]
            //    , localOffsetQueue[i]);

        };

        //sync(cta);


    }



};







// helper functions and utilities to work with CUDA from https://github.com/NVIDIA/cuda-samples



/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYO>
__global__ void getMinMaxes(ForBoolKernelArgs<TYO> fbArgs
    , unsigned int* minMaxes
    , TYO* goldArr, TYO* segmArr, MetaDataGPU metaData
) {

    // __global__ void getMinMaxes(unsigned int* minMaxes) {
     ////////////some initializations
    thread_block cta = this_thread_block();
    //thread_block_tile<32> tile = tiled_partition<32>(cta);



    //shared memory

    __shared__ bool anyInGold[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ unsigned int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) { minMaxesInShmem[1] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 0)) { minMaxesInShmem[2] = 1000; };

    if ((threadIdx.x == 3) && (threadIdx.y == 0)) { minMaxesInShmem[3] = 0; };
    if ((threadIdx.x == 4) && (threadIdx.y == 0)) { minMaxesInShmem[4] = 1000; };

    if ((threadIdx.x == 5) && (threadIdx.y == 0)) { minMaxesInShmem[5] = 0; };
    if ((threadIdx.x == 6) && (threadIdx.y == 0)) { minMaxesInShmem[6] = 1000; };

    if ((threadIdx.x == 7) && (threadIdx.y == 0)) { anyInGold[1] = false; };


    //if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
    //    //printf("in minMaxes beg  totalMetaLength  %d Nx %d Ny %d Nz %d \n"
    //    //    , fbArgs.metaData.totalMetaLength
    //    //    , fbArgs.goldArr.Nx
    //    //    , fbArgs.goldArr.Ny
    //    //    , fbArgs.goldArr.Nz
    //    //
    //    //);

    //    if (blockIdx.x == 0) {
    //        printf(" dims meta in min maxes  kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ \n "
    //            , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
    //            , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
    //            , metaData.minX, metaData.minY, metaData.minZ
    //        );

    //}

    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int  xMeta = linIdexMeta % metaData.metaXLength;
        int   zMeta = int(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * metaData.MetaYLength))));
        int   yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //iterating over data block
        for (uint8_t xLoc = threadIdx.x; xLoc < 32; xLoc += blockDim.x) {
            uint32_t x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = yMeta * fbArgs.dbYLength + yLoc;//absolute position
                //if (y == 0) {
                //    printf("x %d  in min maxes \n ", x);

                //}
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nx) {

                    // resetting 


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nz) {
                            //first array gold
                            //uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                            // setting bits
                            bool goldBool = goldArr[x + y * fbArgs.goldArr.Nx + z * fbArgs.goldArr.Nx * fbArgs.goldArr.Ny] == fbArgs.numberToLookFor;  // (getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            bool segmBool = segmArr[x + y * fbArgs.goldArr.Nx + z * fbArgs.goldArr.Nx * fbArgs.goldArr.Ny] == fbArgs.numberToLookFor;
                            if (goldBool || segmBool) {
                                anyInGold[0] = true;
                                //printf(" \n in min maxes dims meta in min maxes   x %d y%d z%d xMeta %d yMeta %d zMeta %d  kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ %d linIdexMeta %d counted %d  \n "
                                //    ,x,y,z,
                                //    xMeta,yMeta,zMeta
                                //    , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
                                //    , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
                                //    , metaData.minX, metaData.minY, metaData.minZ
                                //    , linIdexMeta
                                //    , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                                //);

                            }



                        }

                    }
                }

                //  __syncthreads();
                  //waiting so shared memory will be loaded evrywhere
                  //on single thread we do last sum reduction

                  /////////////////// setting min and maxes
  //    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
                auto active = coalesced_threads();
                sync(cta);
                active.sync();

                if ((threadIdx.x == 0) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                if ((threadIdx.x == 1) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                if ((threadIdx.x == 2) && (threadIdx.y == 0) && anyInGold[0]) {

                    minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]);

                    //if (minMaxesInShmem[3] > 0) {
                    //    printf(" prim minMaxesInShmem maxY %d meta %d \n ", minMaxesInShmem[3], yMeta);
                    //}

                };
                if ((threadIdx.x == 3) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                if ((threadIdx.x == 4) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                if ((threadIdx.x == 5) && (threadIdx.y == 0) && anyInGold[0]) {
                    minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]);
                    // printf("local fifth %d  \n", minMaxesInShmem[6]);
                };
                // active.sync();
                sync(cta); // just to reduce the warp divergence
                anyInGold[0] = false;




            }
        }

    }
    sync(cta);

    auto active = coalesced_threads();

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        //  printf("\n in minMaxes internal  %d \n", minMaxesInShmem[1]);
       //getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, fbArgs.metaData.minMaxes.Ny, 0, 0)[0] = 61;
        atomicMax(&minMaxes[1], minMaxesInShmem[1]);
        //atomicMax(&minMaxes[1], 2);
       // minMaxes[1] = 0;
    };

    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {

        atomicMin(&minMaxes[2], minMaxesInShmem[2]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[3], minMaxesInShmem[3]);
        //  printf(" minMaxesInShmem maxY %d \n ", minMaxes[3]);

    };

    if ((threadIdx.x == 2) && (threadIdx.y == 0)) {
        atomicMin(&minMaxes[4], minMaxesInShmem[4]);
        //   printf(" minMaxesInShmem minY %d \n ", minMaxes[4]);

    };



    if (threadIdx.x == 3 && threadIdx.y == 0) {
        atomicMax(&minMaxes[5], minMaxesInShmem[5]);
        //  printf(" minMaxesInShmem  %d \n ", minMaxes[5]);
    };

    if (threadIdx.x == 4 && threadIdx.y == 0) {
        atomicMin(&minMaxes[6], minMaxesInShmem[6]);
        // printf(" minMaxesInShmem  %d \n ", minMaxes[6]);

    };





}


























namespace {

    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> input,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> output) {
        //batch index
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        output[c] = input[c]*2;
    }

} // namespace

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor output) {
 
    const int threads = 1024;
    const dim3 blocks(10);


    // from https://github.com/pytorch/pytorch/blob/61d6c4386459441710fb4cfa2929a3f77e95e5f7/aten/src/ATen/Dispatch.h
    AT_DISPATCH_ALL_TYPES(input.type(), "lltm_forward_cuda", ([&] {
            lltm_cuda_forward_kernel<scalar_t> << <blocks, threads >> > (
                input.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                output.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
            }));


    return { input,output };
}

