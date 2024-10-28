/*
Implementations based on:
Harish and Narayanan. "Accelerating large graph algorithms on the GPU using CUDA." HiPC, 2007.
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

/*
Implementations based on:
Harish and Narayanan. "Accelerating large graph algorithms on the GPU using CUDA." HiPC, 2007.
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Terminology (but not values) from graph500 spec
//   graph density = 2^-(2*SCALE - EDGE_FACTOR)
#define SCALE 8
#define EDGE_FACTOR 16

#define N_NODES (1LL << SCALE)
#define N_EDGES (N_NODES * EDGE_FACTOR)

// upper limit
#define N_LEVELS 10

// Larger than necessary for small graphs, but appropriate for large ones
typedef uint64_t edge_index_t;
typedef uint64_t node_index_t;

typedef struct edge_t_struct {
    // These fields are common in practice, but we elect not to use them.
    // weight_t weight;
    // node_index_t src;
    node_index_t dst;
} edge_t;

typedef struct node_t_struct {
    edge_index_t edge_begin;
    edge_index_t edge_end;
} node_t;

typedef int8_t level_t;
#define MAX_LEVEL INT8_MAX

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES], node_index_t starting_node, level_t level[N_NODES], edge_index_t level_counts[N_LEVELS]) {
    node_index_t n;
    edge_index_t e;
    level_t horizon;
    edge_index_t cnt;

    level[starting_node] = 0;
    level_counts[0] = 1;

    loop_horizons:for (horizon = 0; horizon < N_LEVELS; horizon++) {
        #pragma HLS TRIPCOUNT AVG=10
        cnt = 0;
        // Add unmarked neighbors of the current horizon to the next horizon
        loop_nodes:for (n = 0; n < N_NODES; n++) {
            #pragma HLS TRIPCOUNT AVG=64
            if (level[n] == horizon) {
                edge_index_t tmp_begin = nodes[n].edge_begin;
                edge_index_t tmp_end = nodes[n].edge_end;

                loop_neighbours:for (e = tmp_begin; e < tmp_end; e++) {
                    #pragma HLS TRIPCOUNT AVG=16
                    node_index_t tmp_dst = edges[e].dst;
                    level_t tmp_level = level[tmp_dst];

                    if (tmp_level == MAX_LEVEL) { // Unmarked
                        level[tmp_dst] = horizon + 1;
                        ++cnt;
                    }
                }
            }
        }
        if ((level_counts[horizon + 1] = cnt) == 0)
            break;
    }
}
