#include <cstdint>
#define SIZE 2048
#define TYPE int32_t
#define TYPE_MAX INT32_MAX


void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    merge_label1:for(i=start; i<=m; i++){
        #pragma HLS TRIPCOUNT AVG=5.5
        temp[i] = a[i];
    }

    merge_label2:for(j=m+1; j<=stop; j++){
        #pragma HLS TRIPCOUNT AVG=7.5
        temp[m+1+stop-j] = a[j];
    }

    i = start;
    j = stop;

    merge_label3:for(k=start; k<=stop; k++){
        #pragma HLS TRIPCOUNT AVG=12.0
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        if(tmp_j < tmp_i) {
            a[k] = tmp_j;
            j--;
        } else {
            a[k] = tmp_i;
            i++;
        }
    }
}