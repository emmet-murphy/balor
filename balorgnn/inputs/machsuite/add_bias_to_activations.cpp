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
    loop_1:for (i = 0; i < size; i++) {
        #pragma HLS TRIPCOUNT AVG=64
        activations[i] = activations[i] + biases[i];
    }
}