#include <stdlib.h>
#include <stdio.h>

#define NNZ 1666
#define N 494
#define L 10

#define TYPE double

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N]){
    int i, j;
    TYPE Si;

    ellpack_1:for (i=0; i<N; i++) {
        #pragma HLS TRIPCOUNT AVG=494
        TYPE sum = out[i];
        ellpack_2:for (j=0; j<L; j++) {
            #pragma HLS TRIPCOUNT AVG=10
            Si = nzval[j + i*L] * vec[cols[j + i*L]];
            sum += Si;
        }
        out[i] = sum;
    }
}