/*
Implementation based on algorithm described in:
The cache performance and optimizations of blocked algorithms
M. D. Lam, E. E. Rothberg, and M. E. Wolf
ASPLOS 1991
*/

#include <stdio.h>
#include <stdlib.h>
//Data Type
#define TYPE int

//Algorithm Parameters
#define row_size 64
#define col_size 64
#define N row_size*col_size
#define block_size 8


void bbgemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]){
    int i, k, j, jj, kk;
    int i_row, k_row;
    TYPE temp_x, mul;

    loopjj:for (jj = 0; jj < row_size; jj += block_size){
        #pragma HLS TRIPCOUNT AVG=8
        loopkk:for (kk = 0; kk < row_size; kk += block_size){
            #pragma HLS TRIPCOUNT AVG=8
            loopi:for ( i = 0; i < row_size; ++i){
                #pragma HLS TRIPCOUNT AVG=64
                loopk:for (k = 0; k < block_size; ++k){
                    #pragma HLS TRIPCOUNT AVG=8
                    i_row = i * row_size;
                    k_row = (k  + kk) * row_size;
                    temp_x = m1[i_row + k + kk];
                    
                    loopj:for (j = 0; j < block_size; ++j){
                        #pragma HLS TRIPCOUNT AVG=8
                        mul = temp_x * m2[k_row + j + jj];
                        prod[i_row + j + jj] += mul;
                    }
                }
            }
        }
    }
}