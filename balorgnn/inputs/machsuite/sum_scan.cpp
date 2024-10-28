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

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE]) {
    int radixID, bucket_indx;
    sum[0] = 0;
    sum_1:for (radixID = 1; radixID < SCAN_RADIX; radixID++) {
        #pragma HLS TRIPCOUNT AVG=127
        bucket_indx = radixID * SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID - 1] + bucket[bucket_indx];
    }
}