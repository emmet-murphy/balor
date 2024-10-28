
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

#define BUFFER_A 0
#define BUFFER_B 1

void local_scan(int bucket[BUCKETSIZE]) {
    int radixID, i, bucket_indx;
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        for (i = 1; i < SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx - 1];
        }
    }
}

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE]) {
    int radixID, bucket_indx;
    sum[0] = 0;
    for (radixID = 1; radixID < SCAN_RADIX; radixID++) {
        bucket_indx = radixID * SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID - 1] + bucket[bucket_indx];
    }
}

void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX]) {
    int radixID, i, bucket_indx;
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        for (i = 0; i < SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
        }
    }
}

void init(int bucket[BUCKETSIZE]) {
    int i;
    for (i = 0; i < BUCKETSIZE; i++) {
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp) {
    int blockID, i, bucket_indx, a_indx;
    blockID = 0;
    for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        for (i = 0; i < 4; i++) {
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            bucket_indx = ((a[a_indx] >> exp) & 0x3) * NUMOFBLOCKS + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp) {
    int i, blockID, bucket_indx, a_indx;
    blockID = 0;

    for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        for (i = 0; i < 4; i++) {
            bucket_indx = ((a[blockID * ELEMENTSPERBLOCK + i] >> exp) & 0x3) * NUMOFBLOCKS + blockID;
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            b[bucket[bucket_indx]] = a[a_indx];
            bucket[bucket_indx]++;
        }
    }
}

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]) {
    int exp = 0;
    int valid_buffer = 0;

    sort_1:for (exp = 0; exp < 32; exp += 2) {
        init(bucket);
        if (valid_buffer == BUFFER_A) {
            hist(bucket, a, exp);
        } else {
            hist(bucket, b, exp);
        }

        local_scan(bucket);
        sum_scan(sum, bucket);
        last_step_scan(bucket, sum);

        if (valid_buffer == BUFFER_A) {
            update(b, bucket, a, exp);
            valid_buffer = BUFFER_B;
        } else {
            update(a, bucket, b, exp);
            valid_buffer = BUFFER_A;
        }
    }
    // If trip count is even, buffer A will be valid at the end.
}