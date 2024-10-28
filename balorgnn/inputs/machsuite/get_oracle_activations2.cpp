#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Fixed parameters
#define input_dimension 13
#define possible_outputs 3
#define training_sets 163
#define nodes_per_layer 64
#define layers 2
#define learning_rate 0.01
#define epochs 1
#define test_sets 15
#define norm_param 0.005

#define max 1.0
#define offset 0.5

// Data Bounds
#define TYPE double
#define MAX 1000
#define MIN 1

void get_oracle_activations2(TYPE weights3[nodes_per_layer*possible_outputs], TYPE output_differences[possible_outputs], TYPE oracle_activations[nodes_per_layer], TYPE dactivations[nodes_per_layer]) {
    int i, j;
    loop_1:for( i = 0; i < nodes_per_layer; i++) {
        #pragma HLS TRIPCOUNT AVG=64
        oracle_activations[i] = (TYPE)0.0;
        loop_2:for( j = 0; j < possible_outputs; j++) {
            #pragma HLS TRIPCOUNT AVG=3
            oracle_activations[i] += output_differences[j] * weights3[i*possible_outputs + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}
