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

void take_difference(TYPE net_outputs[possible_outputs], TYPE solutions[possible_outputs], TYPE output_difference[possible_outputs], TYPE dactivations[possible_outputs]) {
    int i;
    loop_1:for (i = 0; i < possible_outputs; i++) {
        output_difference[i] = (((net_outputs[i]) - solutions[i]) * -1.0) * dactivations[i];
    }
}