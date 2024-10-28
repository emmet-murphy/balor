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

void local_scan(int bucket[BUCKETSIZE]){
    int radixID, i, bucket_indx;
    local_1:for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        local_2:for (i=1; i<SCAN_BLOCK; i++){
            bucket_indx = radixID*SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx-1];
        }
    }
}