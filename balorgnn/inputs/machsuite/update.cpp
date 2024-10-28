
/*
Implementation based on algorithm described in:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TYPE int32_t
#define TYPE_MAX INT32_MAX

#define SIZE 2048
#define NUMOFBLOCKS 512

#define ELEMENTSPERBLOCK 4
#define RADIXSIZE 4
#define BUCKETSIZE NUMOFBLOCKS *RADIXSIZE
#define MASK 0x3

#define SCAN_BLOCK 16
#define SCAN_RADIX BUCKETSIZE / SCAN_BLOCK

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp) {
    int i, blockID, bucket_indx, a_indx;
    blockID = 0;

    update_1:for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        #pragma HLS TRIPCOUNT AVG=512
        update_2:for (i = 0; i < 4; i++) {
            #pragma HLS TRIPCOUNT AVG=4
            bucket_indx = ((a[blockID * ELEMENTSPERBLOCK + i] >> exp) & 0x3) * NUMOFBLOCKS + blockID;
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            b[bucket[bucket_indx]] = a[a_indx];
            bucket[bucket_indx]++;
        }
    }
}
