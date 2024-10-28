#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

void soft_max(TYPE net_outputs[possible_outputs], TYPE activations[possible_outputs]) {
    int i;
    TYPE sum;
    sum = (TYPE)0.0;

    loop_1:for (i = 0; i < possible_outputs; i++) {
        sum += exp(-activations[i]);
    }
    loop_2:for (i = 0; i < possible_outputs; i++) {
        net_outputs[i] = exp(-activations[i]) / sum;
    }
}