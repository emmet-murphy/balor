#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TYPE int32_t
#define TYPE_MAX INT32_MAX

#define SIZE 2048
#define NUMOFBLOCKS 512

#define ELEMENTSPERBLOCK 4
#define RADIXSIZE 4
#define BUCKETSIZE NUMOFBLOCKS*RADIXSIZE
#define MASK 0x3

#define SCAN_BLOCK 16
#define SCAN_RADIX BUCKETSIZE/SCAN_BLOCK


void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int radixID, i, bucket_indx;
    last_1:for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        #pragma HLS TRIPCOUNT AVG=128
        last_2:for (i=0; i<SCAN_BLOCK; i++) {
            #pragma HLS TRIPCOUNT AVG=16
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
         }
    }
}