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

void update_weights(TYPE weights1[input_dimension * nodes_per_layer], TYPE weights2[nodes_per_layer * nodes_per_layer], TYPE weights3[nodes_per_layer * possible_outputs], TYPE d_weights1[input_dimension * nodes_per_layer], TYPE d_weights2[nodes_per_layer * nodes_per_layer], TYPE d_weights3[nodes_per_layer * possible_outputs], TYPE biases1[nodes_per_layer], TYPE biases2[nodes_per_layer], TYPE biases3[possible_outputs], TYPE d_biases1[nodes_per_layer], TYPE d_biases2[nodes_per_layer], TYPE d_biases3[possible_outputs]) {
    int i, j;
    double norm, bias_norm;
    norm = 0.0;
    bias_norm = 0.0;

    loop_1:for (i = 0; i < input_dimension; i++) {
        loop_1_1:for (j = 0; j < nodes_per_layer; j++) {
            weights1[i * nodes_per_layer + j] -= (d_weights1[i * nodes_per_layer + j] * learning_rate);
            norm += weights1[i * nodes_per_layer + j] * weights1[i * nodes_per_layer + j];
        }
    }
    loop_2:for (i = 0; i < nodes_per_layer; i++) {
        biases1[i] -= (d_biases1[i] * learning_rate);
        bias_norm += biases1[i] * biases1[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    loop_3:for (i = 0; i < input_dimension; i++) {
        loop_3_1:for (j = 0; j < nodes_per_layer; j++) {
            weights1[i * nodes_per_layer + j] = (weights1[i * nodes_per_layer + j] / norm);
        }
    }
    loop_4:for (i = 0; i < nodes_per_layer; i++) {
        biases1[i] = (biases1[i] / bias_norm);
    }

    norm = (double)0.0;
    bias_norm = (double)0.0;

    loop_5:for (i = 0; i < nodes_per_layer; i++) {
        loop_5_1:for (j = 0; j < nodes_per_layer; j++) {
            weights2[i * nodes_per_layer + j] -= (d_weights2[i * nodes_per_layer + j] * learning_rate);
            norm += weights2[i * nodes_per_layer + j] * weights2[i * nodes_per_layer + j];
        }
    }
    loop_6:for (i = 0; i < nodes_per_layer; i++) {
        biases2[i] -= (d_biases2[i] * learning_rate);
        bias_norm += biases2[i] * biases2[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    loop_7:for (i = 0; i < nodes_per_layer; i++) {
        loop_7_1:for (j = 0; j < nodes_per_layer; j++) {
            weights2[i * nodes_per_layer + j] = (weights2[i * nodes_per_layer + j] / norm);
        }
    }
    loop_8:for (i = 0; i < nodes_per_layer; i++) {
        biases2[i] = (biases2[i] / bias_norm);
    }

    norm = (double)0.0;
    bias_norm = (double)0.0;

    loop_9:for (i = 0; i < nodes_per_layer; i++) {
        loop_9_1:for (j = 0; j < possible_outputs; j++) {
            weights3[i * possible_outputs + j] -= (d_weights3[i * possible_outputs + j] * learning_rate);
            norm += weights3[i * possible_outputs + j] * weights3[i * possible_outputs + j];
        }
    }
    loop_10:for (i = 0; i < possible_outputs; i++) {
        biases3[i] -= d_biases3[i] * learning_rate;
        bias_norm += biases3[i] * biases3[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    loop_11:for (i = 0; i < nodes_per_layer; i++) {
        loop_11_1:for (j = 0; j < possible_outputs; j++) {
            weights3[i * possible_outputs + j] = (weights3[i * possible_outputs + j] / norm);
        }
    }
    loop_12:for (i = 0; i < possible_outputs; i++) {
        biases3[i] = (biases3[i] / bias_norm);
    }
}