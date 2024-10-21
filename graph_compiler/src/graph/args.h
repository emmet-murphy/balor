#ifndef BALOR_ARGS_H
#define BALOR_ARGS_H

#include <string>

namespace {
const std::string IGNORE_CONTROL_FLOW_DESC = "Don't add control flow edges to the graph.";
const std::string IGNORE_CALL_EDGES_DESC = "Don't add call edges to the graph.";
const std::string HIDE_VALUES_DESC = "Don't add value nodes to the graph.";
const std::string ABSORB_TYPES_DESC = "Absorb type nodes into individual node embeddings.";
const std::string INLINE_FUNCTIONS_DESC =
    "Inline function calls instead of having function call nodes to a single shared use";
const std::string MAKE_PDF_DESC = "Make a pdf in the outputs folder instead of printing to cout.";
const std::string MAKE_DOT_DESC = "Make a dotfile in the outputs folder instead of printing to cout.";
const std::string REMOVE_SINGLE_TARGET_BRANCHES_DESC =
    "Remove single target branch nodes, such as the ones between different parts of a for loop, or at the end of if "
    "statement bodies.";
const std::string REDUCE_ITERATOR_BITWIDTH_DESC =
    "For iterators with fixed bounds, reduce their bitwidth appropriately";
const std::string ALLOCAS_TO_MEM_ELEMS_DESC = "Use memory element nodes instead of pointer allocas";
const std::string DROP_FUNC_CALL_PROC_DESC =
    "Don't add compiler specific nodes for processing parameters before function calls";
const std::string ABSORB_PRAGMAS_DESC = "Absorb pragma nodes into individual node embeddings";
const std::string PIPELINE_UNROLL_DESC = "When a loop is pipelined, inner loops are automatically unrolled. Mark their average tripcount as their unroll factor.";
const std::string ADD_BB_ID_DESC = "Add BB ID to node embeddings.";
const std::string ADD_FUNC_ID_DESC = "Add function ID to node embeddings.";
const std::string ADD_NODE_TYPE_DESC = "Add node type (instruction, variable, constant, pragma) to embedding.";
const std::string REMOVE_SEXTS_DESC = "Remove sext nodes.";
const std::string ONE_HOT_TYPES_DESC = "One hot encode absorbed types, instead of encoding them descriptively.";
const std::string ADD_EDGE_ORDER_DESC = "Add edge order to the edge embeddings.";
const std::string ONLY_MEMORY_CONTROL_FLOW_DESC =
    "Only add control flow edges between reads, writes, and specify addresses";
const std::string PROXY_PROGRAML_DESC = "Change output details to match PrograML exactly";
const std::string DONT_DISPLAY_TYPES_DESC = "Don't output absorbed type info on the pdf";
const std::string ADD_NUM_CALLS_DESC = "Add the number of calls and call-sites to nodes in sub-functions";
const std::string ADD_SPECIFY_ADDRESS_NODES_DESC = "Add specify address nodes for array reads and writes";
const std::string ADD_EXTERNAL_NODE_DESC = "Add external node to function call graph";
} // namespace

namespace Balor {

const std::string MAKE_PDF = "make_pdf";
const std::string MAKE_DOT = "make_dot";
const std::string HIDE_VALUES = "hide_values";

const std::string IGNORE_CONTROL_FLOW = "ignore_control_flow";
const std::string ABSORB_TYPES = "absorb_types";
const std::string INLINE_FUNCTIONS = "inline_functions";
const std::string IGNORE_CALL_EDGES = "ignore_call_edges";
const std::string REMOVE_SINGLE_TARGET_BRANCHES = "remove_single_target_branches";
const std::string REDUCE_ITERATOR_BITWIDTH = "reduce_iterator_bitwidth";
const std::string ALLOCAS_TO_MEM_ELEMS = "allocas_to_mem_elems";
const std::string DROP_FUNC_CALL_PROC = "drop_func_call_proc";
const std::string ABSORB_PRAGMAS = "absorb_pragmas";
const std::string PIPELINE_UNROLL = "add_unroll_from_pipeline";
const std::string ADD_BB_ID = "add_bb_id";
const std::string ADD_FUNC_ID = "add_func_id";
const std::string ADD_EDGE_ORDER = "add_edge_order";
const std::string ADD_NODE_TYPE = "add_node_type";
const std::string REMOVE_SEXTS = "remove_sexts";
const std::string ONE_HOT_TYPES = "one_hot_types";
const std::string ONLY_MEMORY_CONTROL_FLOW = "only_memory_control_flow";
const std::string PROXY_PROGRAML = "proxy_programl";
const std::string DONT_DISPLAY_TYPES = "no_type_display";
const std::string ADD_NUM_CALLS = "add_num_calls";
const std::string ADD_EXTERNAL_NODE = "add_external";

const std::pair<std::string, std::string> ARGS[] = {
    std::make_pair(IGNORE_CONTROL_FLOW, IGNORE_CONTROL_FLOW_DESC),
    std::make_pair(HIDE_VALUES, HIDE_VALUES_DESC),
    std::make_pair(ABSORB_TYPES, ABSORB_TYPES_DESC),
    std::make_pair(INLINE_FUNCTIONS, INLINE_FUNCTIONS_DESC),
    std::make_pair(IGNORE_CALL_EDGES, IGNORE_CALL_EDGES_DESC),
    std::make_pair(MAKE_PDF, MAKE_PDF_DESC),
    std::make_pair(MAKE_DOT, MAKE_DOT_DESC),
    std::make_pair(REMOVE_SINGLE_TARGET_BRANCHES, REMOVE_SINGLE_TARGET_BRANCHES_DESC),
    std::make_pair(REDUCE_ITERATOR_BITWIDTH, REDUCE_ITERATOR_BITWIDTH_DESC),
    std::make_pair(ALLOCAS_TO_MEM_ELEMS, ALLOCAS_TO_MEM_ELEMS_DESC),
    std::make_pair(DROP_FUNC_CALL_PROC, DROP_FUNC_CALL_PROC_DESC),
    std::make_pair(ABSORB_PRAGMAS, ABSORB_PRAGMAS_DESC),
    std::make_pair(PIPELINE_UNROLL, PIPELINE_UNROLL_DESC),
    std::make_pair(ADD_BB_ID, ADD_BB_ID_DESC),
    std::make_pair(ADD_FUNC_ID, ADD_FUNC_ID_DESC),
    std::make_pair(REMOVE_SEXTS, REMOVE_SEXTS_DESC),
    std::make_pair(ONE_HOT_TYPES, ONE_HOT_TYPES_DESC),
    std::make_pair(ADD_EDGE_ORDER, ADD_EDGE_ORDER_DESC),
    std::make_pair(ONLY_MEMORY_CONTROL_FLOW, ONLY_MEMORY_CONTROL_FLOW_DESC),
    std::make_pair(PROXY_PROGRAML, PROXY_PROGRAML_DESC),
    std::make_pair(DONT_DISPLAY_TYPES, DONT_DISPLAY_TYPES_DESC),
    std::make_pair(ADD_NODE_TYPE, ADD_NODE_TYPE_DESC),
    std::make_pair(ADD_NUM_CALLS, ADD_NUM_CALLS_DESC),
    std::make_pair(ADD_EXTERNAL_NODE, ADD_EXTERNAL_NODE_DESC)
    };
} // namespace Balor

#endif