﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define H5_BUILT_AS_DYNAMIC_LIB 1
#include <H5Cpp.h>


// Copyright (c) MONAI Consortium
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <vector>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cmath>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <assert.h>
#include "device_launch_parameters.h"
#include "h5stream.hpp"


using namespace cooperative_groups;


#pragma once

//constants describing the meaning of main shared memory spaces
constexpr uint32_t localWorkQueLength = 300;
constexpr uint32_t startOfLocalWorkQ = 4160;
constexpr uint32_t lengthOfMainShmem = 4460;//4460;
constexpr uint32_t begResShmem = 1088;
constexpr uint32_t begfirstRegShmem = 2112;
constexpr uint32_t begSecRegShmem = 3136;
constexpr uint32_t begSourceShmem = 32;


//added to linear index meta in order to  mark weather block is of type gold or not
constexpr uint32_t  isGoldOffset = (UINT16_MAX * 10);



/***************************************
 * structs
 * ********************************/


 /**
 In order to be able to use cuda malloc 3d we will implemnt it as a series
 of 3d arrays
 */
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
    unsigned int minMaxes[20];
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
    TFF* goldArr;
    TFF* segmArr;

    int Nx;
    int Ny;
    int Nz;

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

    int* resultListPointerMeta;
    int* resultListPointerLocalCPU;
    int* resultListPointerIterNumb;

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
    TFB* goldArr;
    TFB* segmArr;
    TFB numberToLookFor;
    int Nx;
    int Ny;
    int Nz;

    int* resultListPointerMeta;
    int* resultListPointerLocal;
    int* resultListPointerIterNumb;

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


###main arrays
0-x : reducedGold
(x+1) - 2x : reducedSegm
*/


    float robustnessPercent = 1.0;// 0.95;

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



/***************************************
 * utils
 * ********************************/


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

    return res;

}

/***************************************
 * utils
 * ********************************/

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
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPI>
inline __device__ uint32_t* getSourceReduced(const ForBoolKernelArgs<TXPI>& fbArgs, const int(&iterationNumb)[1]) {


    if ((iterationNumb[0] & 1) == 0) {
        return fbArgs.mainArrAPointer;

    }
    else {
        return fbArgs.mainArrBPointer;
    }


}


/*
gettinng target array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPPI>
inline __device__ uint32_t* getTargetReduced(const ForBoolKernelArgs<TXPPI>& fbArgs, const  int(&iterationNumb)[1]) {

    if ((iterationNumb[0] & 1) == 0) {

        return fbArgs.mainArrBPointer;

    }
    else {
        return fbArgs.mainArrAPointer;

    }

}


/*
dilatation up and down - using bitwise operators
*/
#pragma once
inline __device__ uint32_t bitDilatate(const uint32_t& x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0
*/
#pragma once
inline __device__ uint32_t isBitAt(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}




/***************************************
 * MinMaxes kernel
 * ********************************/
 /*

 iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
 */
#pragma once
template <typename TYO>
__global__ void getMinMaxes(ForBoolKernelArgs<TYO> fbArgs
    , unsigned int* minMaxes
    , TYO* goldArr
    , TYO* segmArr
    , MetaDataGPU metaData
) {

    thread_block cta = this_thread_block();

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


    if ((threadIdx.x == 1) && (threadIdx.y == 0) && (blockIdx.x == 0)) {

    }

    __syncthreads();

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

                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting
                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //first array gold
                            //uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                            // setting bits
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;  // (getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            if (goldBool || segmBool) {
                                anyInGold[0] = true;
                            }
                        }
                    }
                }

                //  __syncthreads();
                  //waiting so shared memory will be loaded evrywhere
                  //on single thread we do last sum reduction

                  /////////////////// setting min and maxes
  //    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
                __syncthreads();

                if ((threadIdx.x == 0) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                if ((threadIdx.x == 1) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                if ((threadIdx.x == 2) && (threadIdx.y == 0) && anyInGold[0]) {

                    minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]);

                };
                if ((threadIdx.x == 3) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                if ((threadIdx.x == 4) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                if ((threadIdx.x == 5) && (threadIdx.y == 0) && anyInGold[0]) {
                    minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]);
                };
                __syncthreads(); // just to reduce the warp divergence
                anyInGold[0] = false;




            }
        }

    }
    __syncthreads();

    auto active = coalesced_threads();

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[1], minMaxesInShmem[1]);
    };

    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {

        atomicMin(&minMaxes[2], minMaxesInShmem[2]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[3], minMaxesInShmem[3]);

    };

    if ((threadIdx.x == 2) && (threadIdx.y == 0)) {
        atomicMin(&minMaxes[4], minMaxesInShmem[4]);
    };



    if (threadIdx.x == 3 && threadIdx.y == 0) {
        atomicMax(&minMaxes[5], minMaxesInShmem[5]);
    };

    if (threadIdx.x == 4 && threadIdx.y == 0) {
        atomicMin(&minMaxes[6], minMaxesInShmem[6]);

    };
}


/***************************************
 * boolPrepareKernel
 * ********************************/


 /*
 iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
 */
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr
    , TYO* goldArr
    , TYO* segmArr
    , unsigned int* minMaxes) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;

    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;

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
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) { anyInSegm[1] = false; };



    __syncthreads();

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
        __syncthreads();
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //char* tensorslice;

                            //first array gold
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
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

                        }
                    }
                }

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
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

        __syncthreads();

        /////adding the block and total number of the Fp's and Fn's
        sumFp = reduce(tile, sumFp, plus<uint32_t>());
        sumFn = reduce(tile, sumFn, plus<uint32_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        __syncthreads();//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();



        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];


            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            localBlockMetaData[1] = sharedForGold[33];

        }
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

        __syncthreads(); // just to reduce the warp divergence


    }



    __syncthreads();


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[8]), fpSFnS[1]);

    };
}




/***************************************
 * firstMetaPrepareKernel
 * ********************************/

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
    __syncthreads();


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {

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


    }
    __syncthreads();
    if ((threadIdx.x == 0)) {
        globalOffsetForBlock[0] = atomicAdd(&(minMaxes[12]), (fpFnLocCounter[0]));

    };
    if ((threadIdx.x == 1)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
        }
    }
    __syncthreads();


    //setting offsets
    for (uint32_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        workQueue[globalWorkQueueCounter[0] + i] = localWorkQueue[i];

        //FP pass
        if (localWorkQueue[i] >= isGoldOffset) {
            metaDataArr[(localWorkQueue[i] - isGoldOffset) * metaData.metaDataSectionLength + 5] = localOffsetQueue[i] + globalOffsetForBlock[0];
        }
        //FN pass
        else {
            metaDataArr[(localWorkQueue[i]) * metaData.metaDataSectionLength + 6] = localOffsetQueue[i] + globalOffsetForBlock[0];

        };


    }



};




/***************************************
 * memory allocations
 * ********************************/


 /*
 Get arguments for kernels
 */
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC>& mainFunArgs
    , int& warpsNumbForMainPass, int& blockForMainPass
    , const int xLen, const int yLen, const int zLen, cudaStream_t stream
) {


    mainFunArgs.Nx = xLen;
    mainFunArgs.Ny = yLen;
    mainFunArgs.Nz = zLen;


    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes, stream);
    ForBoolKernelArgs<TCC> res;

    res.Nx = xLen;
    res.Ny = yLen;
    res.Nz = zLen;

    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    res.metaData.minMaxes = minMaxes;
    res.minMaxes = minMaxes;
    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;

    //printf("in setting bool args ylen %d dbYlen %d calculated meta %d  \n ", yLen, res.dbYLength, int(ceil(yLen / res.dbYLength)));
    res.metaData.metaXLength = int(ceil(xLen / res.dbXLength));
    res.metaData.MetaYLength = int(ceil(yLen / res.dbYLength));;
    res.metaData.MetaZLength = int(ceil(zLen / res.dbZLength));;
    res.metaData.minX = 0;
    res.metaData.minY = 0;
    res.metaData.minZ = 0;
    res.metaData.maxX = res.metaData.metaXLength;
    res.metaData.maxY = res.metaData.MetaYLength;
    res.metaData.maxZ = res.metaData.MetaZLength;

    res.metaData.totalMetaLength = res.metaData.metaXLength * res.metaData.MetaYLength * res.metaData.MetaZLength;
    res.goldArr = mainFunArgs.goldArr;
    res.segmArr = mainFunArgs.segmArr;


    return res;
}


#pragma once
/*
allocate memory after first kernel
*/
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream) {
    ////reduced arrays
    uint32_t* origArr;
    uint32_t* metaDataArr;
    uint32_t* workQueue;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2] + 1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4] + 1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6] + 1;
    unsigned int totalMetaLength = (xRange) * (yRange) * (zRange);


    printf("cpuArgs.metaData.minMaxes[1] %d  ", cpuArgs.metaData.minMaxes[1]);
    printf("cpuArgs.metaData.minMaxes[2] %d  ", cpuArgs.metaData.minMaxes[2]);
    printf("cpuArgs.metaData.minMaxes[3] %d  ", cpuArgs.metaData.minMaxes[3]);
    printf("cpuArgs.metaData.minMaxes[4] %d  ", cpuArgs.metaData.minMaxes[4]);
    printf("cpuArgs.metaData.minMaxes[5] %d  ", cpuArgs.metaData.minMaxes[5]);
    printf("cpuArgs.metaData.minMaxes[6] %d  ", cpuArgs.metaData.minMaxes[6]);
    printf("cpuArgs.metaData.minMaxes[7] %d  ", cpuArgs.metaData.minMaxes[7]);
    printf("cpuArgs.metaData.minMaxes[8] %d  ", cpuArgs.metaData.minMaxes[8]);
    printf("cpuArgs.metaData.minMaxes[9] %d  ", cpuArgs.metaData.minMaxes[9]);



    //updating size informations
    gpuArgs.metaData.metaXLength = xRange;
    gpuArgs.metaData.MetaYLength = yRange;
    gpuArgs.metaData.MetaZLength = zRange;
    gpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    gpuArgs.metaData.maxX = cpuArgs.metaData.minMaxes[1];
    gpuArgs.metaData.minX = cpuArgs.metaData.minMaxes[2];
    gpuArgs.metaData.maxY = cpuArgs.metaData.minMaxes[3];
    gpuArgs.metaData.minY = cpuArgs.metaData.minMaxes[4];
    gpuArgs.metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    gpuArgs.metaData.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    gpuArgs.metaData.mainArrXLength = mainArrXLength;
    gpuArgs.metaData.mainArrSectionLength = mainArrSectionLength;

    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, stream);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, stream);
    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    cudaMallocAsync(&workQueue, sizeC, stream);
    gpuArgs.origArrsPointer = origArr;
    gpuArgs.metaDataArrPointer = metaDataArr;
    gpuArgs.workQueuePointer = workQueue;
    return gpuArgs.metaData;
};




/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream, bool resIterNeeded, bool res3DNeeded) {

    int* resultListPointerMeta;
    int* resultListPointerLocal;
    int* resultListPointerIterNumb;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(int32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, stream);
    cudaMallocAsync(&resultListPointerIterNumb, size, stream);
    cudaMallocAsync(&resultListPointerMeta, size, stream);




    auto xRange = gpuArgs.metaData.metaXLength;
    auto yRange = gpuArgs.metaData.MetaYLength;
    auto zRange = gpuArgs.metaData.MetaZLength;


    size_t sizeB = gpuArgs.metaData.totalMetaLength * gpuArgs.metaData.mainArrSectionLength * sizeof(uint32_t);


    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);

    //just in order set it to 0
    int* resultListPointerMetaCPU = (int*)calloc(fpPlusFn + 50, sizeof(int));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(resultListPointerIterNumb, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    free(resultListPointerMetaCPU);

    gpuArgs.resultListPointerMeta = resultListPointerMeta;
    gpuArgs.resultListPointerLocal = resultListPointerLocal;
    gpuArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    gpuArgs.mainArrAPointer = mainArrAPointer;
    gpuArgs.mainArrBPointer = mainArrBPointer;


    return fpPlusFn;
};







/***************************************
 * main kernel
 * ********************************/
template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {


    printf("main paqss kernel");
    thread_block cta = cooperative_groups::this_thread_block();

    grid_group grid = cooperative_groups::this_grid();

    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-  4127: small 32 length resgister 3 space
    4128-4500 (372 length) : place for local work queue in dilatation kernels
    */
    // __shared__ uint32_t mainShmem[lengthOfMainShmem];
    __shared__ uint32_t mainShmem[lengthOfMainShmem];
    //cuda::associate_access_property(&mainShmem, cuda::access_property::shared{});



    constexpr size_t stages_count = 2; // Pipeline stages number


    //usefull for iterating through local work queue
    __shared__ bool isGoldForLocQueue[localWorkQueLength];
    // holding data about paddings


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ bool isBlockFull[2];

    __shared__ uint32_t lastI[1];


    //variables needed for all threads
    __shared__ int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    __shared__ unsigned int fpFnLocCounter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];


    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
    0 : global FP count;
    1 : global FN count;
    2 : workQueueCounter
    3 : resultFP globalCounter
    4 : resultFn globalCounter
    */
    __shared__ unsigned int localMinMaxes[5];

    /* will be used to store all of block metadata
  nothing at  0 index
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
 12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
 13 : top
 14 : bottom
 15 : left
 16 : right
 17 : anterior
 18 : posterior
    */

    __shared__ uint32_t localBlockMetaData[40];

    /*
 //now linear indexes of the previous block in all sides - if there is no block in given direction it will equal UINT32_MAX
 0 : top
 1 : bottom
 2 : left
 3 : right
 4 : anterior
 5 : posterior
    */


    /////used mainly in meta passes

//    __shared__ unsigned int fpFnLocCounter[1];
    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];





    //initializations and loading
    if (threadIdx.x == 9 && threadIdx.y == 0) { iterationNumb[0] = -1; };
    if (threadIdx.x == 11 && threadIdx.y == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (threadIdx.x == 12 && threadIdx.y == 0) {
        isSegmPassToContinue[0] = true;

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number
    __syncthreads();

    do {

        for (uint8_t isPaddingPass = 0; isPaddingPass < 2; isPaddingPass++) {
            printf(" iteer ");

            /////////////////////////****************************************************************************************************************
              /// dilataions

    //initial cleaning  and initializations include loading min maxes
            if (threadIdx.x == 7 && threadIdx.y == 0 && !isPaddingPass) {
                iterationNumb[0] += 1;
            };

            if (threadIdx.x == 6 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            };

            if (threadIdx.x == 1 && threadIdx.y == 0) {
                blockFpConter[0] = 0;
            };
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                blockFnConter[0] = 0;
            };
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localFpConter[0] = 0;
            };
            if (threadIdx.x == 4 && threadIdx.y == 0) {
                localFnConter[0] = 0;
            };
            if (threadIdx.x == 9 && threadIdx.y == 0) {
                isBlockFull[0] = true;
            };
            if (threadIdx.x == 9 && threadIdx.y == 1) {
                isBlockFull[1] = true;
            };

            if (threadIdx.x == 10 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            };



            if (threadIdx.x == 0 && threadIdx.y == 0) {
                localTotalLenthOfWorkQueue[0] = fbArgs.minMaxes[9];
                globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
                worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
            };

            if (threadIdx.y == 1) {
                cooperative_groups::memcpy_async(cta, (&localMinMaxes[0]), (&fbArgs.minMaxes[7]), (sizeof(unsigned int) * 5));
            }

            __syncthreads();

            /// load work QueueData into shared memory
            for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {

                //grid stride loop - sadly most of threads will be idle
               ///////// loading to work queue
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {

                        mainShmem[startOfLocalWorkQ + ii] = fbArgs.workQueuePointer[bigloop + ii];
                        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
                        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];


                    }

                }
                //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
                ////##### pipeline Step 0

                __syncthreads();




                //loading metadata
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    memcpy_async(cta, (&localBlockMetaData[0]),
                        (&fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ] * fbArgs.metaData.metaDataSectionLength])
                        , (sizeof(uint32_t) * 20));

                }


                __syncthreads();

                for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {




                    if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                        memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                            mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                            (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength));

                        __syncthreads();

                        ///////// step 1 load top and process main data
                                        //load top
                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        //process main
                        //marking weather block is already full and no more dilatations are possible
                        if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) < 32) {
                            isBlockFull[i & 1] = false;
                        }
                        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
                        __syncthreads();

                        ///////// step 2 load bottom and process top
                                        //load bottom
                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        //process top
                        __syncthreads();


                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 0)) {
                                // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                                isAnythingInPadding[0] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32] >> 31) & 1) << 0;
                        }

                        __syncthreads();

                        /////////// step 3 load right  process bottom
                        if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                            memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        //process bototm
                        __syncthreads();


                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 31)) {
                                isAnythingInPadding[1] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] >> 0) & 1) << 31;
                        }



                        /*  dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                              , 0, 31
                              , begSecRegShmem, i);*/

                        __syncthreads();


                        /////////// step 4 load left process right
                                        //load left
                        if (mainShmem[startOfLocalWorkQ + i] > 0) {
                            memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        //process right
                        __syncthreads();

                        if (threadIdx.x == (fbArgs.dbXLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[3] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + (threadIdx.y * 32)];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x + 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        __syncthreads();
                        /////// step 5 load anterior process left
                                        //load anterior
                        if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

                            memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        //process left
                        __syncthreads();

                        // so we first check for corner cases
                        if (threadIdx.x == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 15] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[2] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + 31 + threadIdx.y * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x - 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        __syncthreads();

                        /////// step 6 load posterior process anterior
                                        //load posterior
                        if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


                            memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );
                        }
                        __syncthreads();

                        //process anterior

                        // so we first check for corner cases
                        if (threadIdx.y == (fbArgs.dbYLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[4] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + threadIdx.x];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y + 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        __syncthreads();

                        /////// step 7
                                       //load reference if needed or data for next iteration if there is such
                                        //process posterior, save data from res shmem to global memory also we mark weather block is full

                        //if block should be validated we load data for validation
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                        > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &fbArgs.origArrsPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for
                                (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                            );

                        }
                        else {//if we are not validating we immidiately start loading data for next loop
                            if (i + 1 < worQueueStep[0]) {
                                memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , (sizeof(uint32_t) * 20));


                            }
                        }

                        __syncthreads();


                        //processPosteriorAndSaveResShmem

                        //dilatate posterior


                        // so we first check for corner cases
                        if (threadIdx.y == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[5] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + threadIdx.x + (fbArgs.dbYLength - 1) * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y - 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }
                        __syncthreads();

                        //now all data should be properly dilatated we save it to global memory
                        //try save target reduced via mempcy async ...


                        //memcpy_async(cta,
                        //    &getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])]
                        //    , (&mainShmem[begResShmem]),
                        //     (sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                        //    , pipeline);



                        getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                            + threadIdx.x + threadIdx.y * 32]
                            = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];






                        __syncthreads();

                        //////// step 8 basically in order to complete here anyting the count need to be bigger than counter
                                                      // loading for next block if block is not to be validated it was already done earlier
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            if (i + 1 < worQueueStep[0]) {


                                memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , (sizeof(uint32_t) * 20));

                            }
                        }




                        __syncthreads();

                        //validation - so looking for newly covered voxel for opposite array so new fps or new fns

                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                                        // now we look through bits and when some is set we call it a result
#pragma unroll
                            for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                                //if any bit here is set it means it should be added to result list
                                if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && !isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && isBitAt(mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    ) {

                                    //just re
                                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = 0;
                                    ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                                    if (isGoldForLocQueue[i]) {
                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3]);

                                    }
                                    else {

                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);


                                    };
                                    //   add results to global memory
                                    //we add one gere jjust to distinguish it from empty result
                                    fbArgs.resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                                    fbArgs.resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                                    fbArgs.resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0] + 1);
                                    printf(" reees %d" );



                                }

                            };

                        }
                        /////////

                        /// /// cleaning

                        __syncthreads();

                        if (threadIdx.x == 9 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

         //executed in case of previous block
                            if (isBlockFull[i & 1] && i >= 0) {
                                //setting data in metadata that block is full
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
                            }
                            //resetting for some reason  block 0 gets as full even if it should not ...
                            isBlockFull[i & 1] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
                        };




                        //we do it only for non padding pass
                        if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
                            //executed in case of previous block
                            if (i >= 0) {

                                if (localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] < isGoldOffset) {

                                    if (isAnythingInPadding[threadIdx.x]) {
                                        // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
                                        fbArgs.metaDataArrPointer[localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] * fbArgs.metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                                    }

                                }
                            }
                            isAnythingInPadding[threadIdx.x] = false;
                        };






                        if (threadIdx.x == 7 && threadIdx.y == 0) {
                            //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
                            if (localFpConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 3] += localFpConter[0];

                                blockFpConter[0] += localFpConter[0];
                                localFpConter[0] = 0;
                            }


                        };
                        if (threadIdx.x == 8 && threadIdx.y == 0) {

                            if (localFnConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 4] += localFnConter[0];

                                blockFnConter[0] += localFnConter[0];
                                localFnConter[0] = 0;
                            }
                        };

                        __syncthreads();

                    }
                }

                //here we are after all of the blocks planned to be processed by this block are



            }

            __syncthreads();

            //     updating global counters
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                if (blockFpConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[10]), (blockFpConter[0]));
                }
            };
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                if (blockFnConter[0] > 0) {
                    //if (blockFnConter[0]>10) {
                    //    printf("Fn %d  ", blockFnConter[0]);
                    //}
                    atomicAdd(&(fbArgs.minMaxes[11]), (blockFnConter[0]));
                }
            };

            // in first thread block we zero work queue counter
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                if (blockIdx.x == 0) {

                    fbArgs.minMaxes[9] = 0;
                }
            };

            grid.sync();
            /////////////////////////****************************************************************************************************************

            /// metadata pass


            // preparation loads
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            }
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;

            }

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isGoldPassToContinue[0]
                    = ((fbArgs.minMaxes[7] * fbArgs.robustnessPercent) > fbArgs.minMaxes[10]);

            };

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isSegmPassToContinue[0]
                    = ((fbArgs.minMaxes[8] * fbArgs.robustnessPercent) > fbArgs.minMaxes[11]);
            };


            __syncthreads();

            /////////////////////////////////

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //goldpass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7] = 1;


                    };

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //gold pass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);

                    };

                }
            }

            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
                }
                __syncthreads();

            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {

                localWorkQueueCounter[0] = 0;
            }
            __syncthreads();

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;

                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9] = 1;

                    }

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
                    }

                }
            }
            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];

                }

            }



            grid.sync();
        }



    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);


    //setting global iteration number to local one
    if (blockIdx.x == 0) {
        if (threadIdx.x == 2 && threadIdx.y == 0) {
            fbArgs.metaData.minMaxes[13] = (iterationNumb[0] + 1);
        }
    }
}




/***************************************
 * putting all kernels and memory allocations together
 * ********************************/


 /*
 get data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
 */
template <typename T>
inline occupancyCalcData getOccupancy() {

    occupancyCalcData res;

    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)getMinMaxes<T>,
        0);
    res.warpsNumbForMinMax = blockSize / 32;
    res.blockSizeForMinMax = minGridSize;

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.warpsNumbForboolPrepareKernel = blockSize / 32;
    res.blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.theadsForFirstMetaPass = blockSize;
    res.blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<T>,
        0);
    res.warpsNumbForMainPass = blockSize / 32;
    res.blockForMainPass = minGridSize;

    //printf("warpsNumbForMainPass %d blockForMainPass %d  ", res.warpsNumbForMainPass, res.blockForMainPass);
    return res;
}



/*
TODO consider representing as a CUDA graph
executing Algorithm as CUDA graph  based on official documentation and
https://codingbyexample.com/2020/09/25/cuda-graph-usage/
*/
#pragma once
template <typename T>
ForBoolKernelArgs<T> executeHausdoff(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, occupancyCalcData& occData,
    cudaStream_t stream, bool resToSave, float robustnessPercent, bool resIterNeeded, bool res3DNeeded) {
    fFArgs.robustnessPercent = robustnessPercent;

    printf("executeHausdoff");

    T* goldArrPointer = fFArgs.goldArr;
    T* segmArrPointer = fFArgs.segmArr;

    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, occData.warpsNumbForMainPass, occData.blockForMainPass, WIDTH, HEIGHT, DEPTH, stream);

    getMinMaxes << <occData.blockSizeForMinMax, dim3(32, occData.warpsNumbForMinMax), 0, stream >> > (fbArgs, fbArgs.minMaxes
        , goldArrPointer
        , segmArrPointer
        , fbArgs.metaData);



    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, stream);
    fbArgs.robustnessPercent = robustnessPercent;
    boolPrepareKernel << <occData.blockSizeFoboolPrepareKernel, dim3(32, occData.warpsNumbForboolPrepareKernel), 0, stream >> > (
        fbArgs, fbArgs.metaData, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer
        , goldArrPointer
        , segmArrPointer
        , fbArgs.minMaxes);


    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs, stream, resIterNeeded, res3DNeeded);



    firstMetaPrepareKernel << <occData.blockForFirstMetaPass, occData.theadsForFirstMetaPass, 0, stream >> > (fbArgs, fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer);



    void* kernel_args[] = { &fbArgs };
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), occData.blockForMainPass, dim3(32, occData.warpsNumbForMainPass), kernel_args, 0, stream);

    if (!resIterNeeded || !res3DNeeded) {
        cudaFreeAsync(fbArgs.resultListPointerIterNumb, stream);
    }
    if (!res3DNeeded) {
        cudaFreeAsync(fbArgs.resultListPointerMeta, stream);
        cudaFreeAsync(fbArgs.resultListPointerLocal, stream);
    }
    cudaFreeAsync(fbArgs.workQueuePointer, stream);
    cudaFreeAsync(fbArgs.origArrsPointer, stream);
    cudaFreeAsync(fbArgs.metaDataArrPointer, stream);
    cudaFreeAsync(fbArgs.mainArrAPointer, stream);
    cudaFreeAsync(fbArgs.mainArrBPointer, stream);


    return fbArgs;

}







template <typename T>
int getHausdorffDistance_CUDA_Generic(T* goldStandard, T* algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, bool resIterNeeded, T numberToLookFor, bool res3DNeeded) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor;
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, resIterNeeded, res3DNeeded);

    size_t sizeMinMax = sizeof(unsigned int) * 20;
    //making sure we have all resultsto copy on cpu
    cudaDeviceSynchronize();
    cudaMemcpy(metaData.minMaxes, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    int result = metaData.minMaxes[13];

    cudaFreeAsync(fbArgs.minMaxes, stream1);
    //free(minMaxesCPU);


    cudaStreamDestroy(stream1);
    cudaDeviceSynchronize();

    return result;
}


template <typename T>
ForBoolKernelArgs<T> getHausdorffDistance_CUDA_FullResList_local(T* goldStandard,
    T* algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, T numberToLookFor) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    //forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor;
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, true, false);



    cudaFreeAsync(fbArgs.minMaxes, stream1);
    //free(metaData.minMaxesCPU);



    cudaStreamDestroy(stream1);

    return fbArgs ;




}




/***************************************
 *enable getting localizations of the voxels that contributed to HD
 * ********************************/



 /*
 on the basis of result lists return the location of each voxel that contributed to Hausdorff distance and how much it contributed
 fbArgs - struct with needed data
 resGold - tensor where output will be stored from gold mask dilatations
 resSegm - tensor where output will be stored from algorithm output dilatations
 len - length of result list we will iterate over
 */
template <typename T>
__global__ void get3Dres_local_kernel(ForBoolKernelArgs<T> fbArgs, T* resGold, T* resSegm, int len) {

    //simple grid stride loop
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        if (fbArgs.resultListPointerLocal[i] > 0 || fbArgs.resultListPointerMeta[i] > 0) {
            uint32_t linIdexMeta = fbArgs.resultListPointerMeta[i] - (isGoldOffset * (fbArgs.resultListPointerMeta[i] >= isGoldOffset)) - 1;
            uint32_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
            uint32_t zMeta = uint32_t(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength))));
            uint32_t yMeta = uint32_t(floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength)));

            auto linLocal = fbArgs.resultListPointerLocal[i];
            auto xLoc = linLocal % fbArgs.dbXLength;
            auto zLoc = uint32_t(floor((float)(linLocal / (32 * fbArgs.dbYLength))));
            auto yLoc = uint32_t(floor((float)((linLocal - ((zLoc * 32 * fbArgs.dbYLength) + xLoc)) / 32)));

            // setting appropriate  spot in the result to a given value
            if (fbArgs.resultListPointerMeta[i] >= isGoldOffset) {
                resGold[(xMeta * 32 + xLoc) + (yMeta * fbArgs.dbYLength + yLoc) * fbArgs.Nx + (zMeta * 32 + zLoc) * fbArgs.Nx * fbArgs.Ny] = fbArgs.resultListPointerIterNumb[i]; // krowa
            }
            else {
                resSegm[(xMeta * 32 + xLoc) + (yMeta * fbArgs.dbYLength + yLoc) * fbArgs.Nx + (zMeta * 32 + zLoc) * fbArgs.Nx * fbArgs.Ny] = fbArgs.resultListPointerIterNumb[i]; //krowa
            }
            //uint32_t x = xMeta * 32 + xLoc;
            //uint32_t y = yMeta * fbArgs.dbYLength + yLoc;
            //uint32_t z = zMeta * 32 + zLoc;
            //uint32_t iterNumb = fbArgs.resultListPointerIterNumb[i];

            //printf("resullt linIdexMeta %d x %d y %d z %d  xMeta %d yMeta %d zMeta %d xLoc %d yLoc %d zLoc %d linLocal %d  iterNumb %d \n"
            //    , linIdexMeta
            //    , x, y, z
            //    , xMeta, yMeta, zMeta
            //    , xLoc, yLoc, zLoc
            //    , linLocal
            //    , iterNumb
        }
    }
}


/*
takes two 3D tensord and computes the element wise avarage from two entries and save result in resGold
voxelsNumber - number of voxel in resGold = resSegm
*/
template <typename T>
__global__ void elementWiseAverage(ForBoolKernelArgs<T> fbArgs, T* resGold, T* resSegm, int voxelsNumber) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelsNumber; i += blockDim.x * gridDim.x) {
        resGold[i] = (resGold[i] + resSegm[i]) / 2;
    }
}




/*
3D tensor with data how much given voxel contributed to result in gold mask segmentations  other mask dilatations (the mean of those)
*/
template <typename T>
float* getHausdorffDistance_CUDA_3Dres_local(T* goldStandard,
    T* algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, T numberToLookFor) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor;
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, true, true);



    size_t sizeRes = sizeof(float) * WIDTH * HEIGHT * DEPTH;
    float* resGold;
    float* resSegm;
    cudaMallocAsync(&resGold, sizeRes, stream);
    cudaMallocAsync(&resSegm, sizeRes, stream);

    unsigned int fpPlusFn = forFullBoolPrepArgs.metaData.minMaxes[7] + forFullBoolPrepArgs.metaData.minMaxes[8];
    int len = (fpPlusFn + 50);
    //occupancy calculator
    int minGridSize = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)get3Dres_local_kernel<T>,
        0);

    //simple one dimensional kernel
    get3Dres_local_kernel << <minGridSize, blockSize, 0, stream1 >> > (fbArgs, resGold, resSegm, len);

    //get element wise average
    elementWiseAverage << <minGridSize, blockSize, 0, stream1 >> > (fbArgs, resGold, resSegm, WIDTH * HEIGHT * DEPTH);


    cudaFreeAsync(fbArgs.minMaxes, stream1);

    cudaFreeAsync(fbArgs.resultListPointerIterNumb, stream1);

    cudaFreeAsync(fbArgs.resultListPointerMeta, stream1);
    cudaFreeAsync(fbArgs.resultListPointerLocal, stream1);
    cudaFreeAsync(resSegm, stream1);



    cudaStreamDestroy(stream1);

    return  resGold;

}



void loadHDFIntoBoolArr(H5std_string FILE_NAME, H5std_string DATASET_NAME, uint8_t*& data) {

    H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
    //H5::Group tmpGroup = file.openGroup("0");
    H5::DataSet dset = file.openDataSet(DATASET_NAME);
    /*
     * Get the class of the datatype that is used by the dataset.
     */
    H5T_class_t type_class = dset.getTypeClass();
    H5::DataSpace dspace = dset.getSpace();
    int rank = dspace.getSimpleExtentNdims();

    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims, NULL); // rank = 1
    printf("Datasize: %d \n ", dims[0]); // this is the correct number of values

    // Define the memory dataspace
    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    H5::DataSpace memspace(1, dimsm);
    data = (uint8_t*)calloc(dims[0], sizeof(uint8_t));
    dset.read(data, H5::PredType::NATIVE_UINT8, memspace, dspace);
    file.close();

}


void loadHDF() {
    //336×250×371

    const int WIDTH = 336;
    const int HEIGHT = 250;
    int DEPTH = 371;

    const H5std_string FILE_NAMEonlyLungsBoolFlat("D:\\data\\hdf5Data\\smallLiverDataSet.hdf5");
    const H5std_string FILE_NAMEonlyBladderBoolFlat("D:\\data\\hdf5Data\\smallLiverDataSet.hdf5");

    const H5std_string DATASET_NAMEonlyLungsBoolFlat("algoOuttD");
    //const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlatB");
    // create a vector the same size as the dataset
    uint8_t* onlyLungsBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlat, DATASET_NAMEonlyLungsBoolFlat, onlyLungsBoolFlat);



    const H5std_string DATASET_NAMEonlyBladderBoolFlat("golddD");
    //const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlatB");
    // create a vector the same size as the dataset
    uint8_t* onlyBladderBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlat, onlyBladderBoolFlat);

    ForBoolKernelArgs<uint8_t> fbArgs = getHausdorffDistance_CUDA_FullResList_local(onlyLungsBoolFlat, onlyBladderBoolFlat, WIDTH, HEIGHT, DEPTH, 1.0, uint8_t(1) );

     //int* 
    cudaDeviceSynchronize();

    unsigned int* locMinMaxes= (unsigned int*)calloc(20, sizeof(unsigned int));

    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(fbArgs.minMaxes, locMinMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = locMinMaxes[7] + locMinMaxes[8];
    int len = (fpPlusFn + 50);
    int* locRes = (int*)calloc(len, sizeof(int));
    size = sizeof(int) * len;
    cudaMemcpy(locRes, fbArgs.resultListPointerIterNumb, size, cudaMemcpyDeviceToHost);


    

    //1 :fpCount
    //    2 : fnCount
    //    3 : fpCounter
    //    4 : fnCounter
    //    5 : fpOffset
    //    6 : fnOffset
    //    7 : isActiveGold
    //    8 : isFullGold
    //    9 : isActiveSegm
    //    10 : isFullSegm
    //    11 : isToBeActivatedGold
    //    12 : isToBeActivatedSegm
    //    12 : isToBeActivatedSegm
    printf(" loop ,");

    for (int i = 1; i < 13;i++) 
    {
        printf("  value %d index %d   ", locMinMaxes[i], i);
    
    }


   // printf("%d",len);
   //H5::H5File file(FILE_NAMEonlyLungsBoolFlat, H5F_ACC_RDWR);
   ////H5::Group tmpGroup = file.openGroup("0");
   //hsize_t dimsf[1] = { len };
   // H5::DataSpace dataspace(1, dimsf);
   // H5::DataSet dataset = file.createDataSet("res_data_set", H5::PredType::NATIVE_INT, dataspace);
   // dataset.write(locRes, H5::PredType::NATIVE_INT);


    // from https://github.com/srbhp/h5stream


    //std::vector<int> lecResVect(locRes, locRes + len);

    ////file.
    //h5stream::h5stream file(FILE_NAMEonlyLungsBoolFlat, "tr");
    ////copy data to vector
    //std::vector<double> matrix{ 1, 2, 3282, 932 };
    //file.write<double>(matrix, "matrix");

    //file.close();
    free(locMinMaxes);
    cudaFree(fbArgs.resultListPointerIterNumb);








    //onlyBladderBoolFlat = (bool*)calloc(WIDTH* HEIGHT* DEPTH, sizeof(bool));

    //onlyBladderBoolFlat[0] = true;

    ////benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//125 
    ////benchmarkMitura(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//124 or 259
    //benchmarkMitura(onlyLungsBoolFlat, onlyBladderBoolFlat, WIDTH, HEIGHT, DEPTH);//124 or 259


    //DEPTH = 536;
    //const H5std_string FILE_NAMEonlyLungsBoolFlatB("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    //const H5std_string DATASET_NAMEonlyLungsBoolFlatB("onlyLungsBoolFlatB");
    //// create a vector the same size as the dataset
    //bool* onlyLungsBoolFlatB;
    //loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlatB, DATASET_NAMEonlyLungsBoolFlatB, onlyLungsBoolFlatB);


    //const H5std_string DATASET_NAMEonlyBladderBoolFlatB("onlyBladderBoolFlatB");
    //// create a vector the same size as the dataset
    //bool* onlyBladderBoolFlatB;
    //loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlatB, onlyBladderBoolFlatB);

    ////benchmarkOliviera(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//125 
    //benchmarkMitura(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//124 or 259

}

//
//
//
//void testAll() {
//
//
//    const int WIDTH = 512;
//    const int HEIGHT = 512;
//    //    const int DEPTH = 536;
//
//    int DEPTH = 826;
//
//
//    //// some preparations and configuring
//    MetaDataCPU metaData;
//    size_t size = sizeof(unsigned int) * 20;
//    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
//    metaData.minMaxes = minMaxesCPU;
//
//    bool* arrA = (bool*)calloc(WIDTH * HEIGHT * DEPTH, sizeof(bool));
//    bool* arrB = (bool*)calloc(WIDTH * HEIGHT * DEPTH, sizeof(bool));
//
//
//    arrA[0] = true;
//    arrB[500] = true;
//
//    ForFullBoolPrepArgs<bool> forFullBoolPrepArgs;
//    forFullBoolPrepArgs.metaData = metaData;
//    forFullBoolPrepArgs.numberToLookFor = true;
//    forFullBoolPrepArgs.goldArr = get3dArrCPU(arrA, WIDTH, HEIGHT, DEPTH);
//    forFullBoolPrepArgs.segmArr = get3dArrCPU(arrB, WIDTH, HEIGHT, DEPTH);
//
//    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH);
//    free(arrA);
//    free(arrB);
//
//}



int main(void) {


    loadHDF();
    // testAll();


    return 0;  // successfully terminated
}
