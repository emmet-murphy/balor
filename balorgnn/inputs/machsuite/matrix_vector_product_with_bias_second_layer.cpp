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


void add_bias_to_activations(TYPE biases[nodes_per_layer], TYPE activations[nodes_per_layer], int size) {
    int i;
    for (i = 0; i < size; i++) {
        activations[i] = activations[i] + biases[i];
    }
}

void matrix_vector_product_with_bias_second_layer(TYPE biases[nodes_per_layer], TYPE weights[nodes_per_layer * nodes_per_layer], TYPE activations[nodes_per_layer], TYPE input_activations[nodes_per_layer]) {
    int i, j;
    loop_1:for (i = 0; i < nodes_per_layer; i++) {
        #pragma HLS TRIPCOUNT AVG=64
        activations[i] = (TYPE)0.0;
        loop_1_1:for (j = 0; j < nodes_per_layer; j++) {
            #pragma HLS TRIPCOUNT AVG=64
            activations[i] += weights[i * nodes_per_layer + j] * input_activations[j];
        }
    }
    add_bias_to_activations(biases, activations, nodes_per_layer);
}
