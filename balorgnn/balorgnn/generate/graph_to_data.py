import numpy as np
import torch

import subprocess
import pygraphviz as pgv

import os


from balorgnn.generate.graph_config import EncoderMethod, EncoderType

from collections import defaultdict



def edges_list_to_edge_array(edges, nodes):
    # get a dict mapping each node name to its index in the node list
    index_map = {value: index for index, value in enumerate(nodes)}

    # separate list of tuples into two lists
    source_names, destination_names = zip(*edges)
    
    # turn names of nodes into node indices
    sources = [index_map[node_name] for node_name in source_names]
    destinations = [index_map[node_name] for node_name in destination_names]


    # bidirectional edges: concatenate each list with the other list
    biSource = np.concatenate((sources, destinations))
    biDestinations = np.concatenate((destinations, sources))

    # make a (2,N) array
    coo = np.array([biSource, biDestinations], dtype=np.int64)

    return torch.from_numpy(coo)

def preprocess_one_hot_attr(obj, attr):
    if obj.attr[attr] is None:
        print(f"node {obj.attr['keyText']} did not have label: ", attr)
        quit()
    attr = obj.attr[attr]
    attr_without_whitespace = attr.replace(" ", "")
    return attr_without_whitespace

def preprocess_normalize_attr(obj, attr):
    if obj.attr[attr] is None:
        print(f"node {obj.attr['keyText']} did not have label: ", attr)
        print()
        quit()
    attr = obj.attr[attr]
    float_attr = float(attr)
    return float_attr


def get_attr_array(encoders, obj_list, arrayType):
    encoded_list = []
    for obj in obj_list:
        features = []
        for encoder in encoders:
            if encoder.type == arrayType:
                if encoder.method == EncoderMethod.ONE_HOT:
                    input = preprocess_one_hot_attr(obj, encoder.label)
                    try:
                        one_hot = encoder.encode(input)
                    except Exception as e:     
                        modified_exception = ValueError(f"There was an error in one hot encoding {input} for {encoder.label}")
                        raise modified_exception from e            
                    features.append(one_hot)
                if encoder.method == EncoderMethod.NORMALIZED:
                    input = preprocess_normalize_attr(obj, encoder.label)
                    normalized_input = encoder.normalize(input)
                    features.append(normalized_input)

        features = np.concatenate(features)
        encoded_list.append(features)
    
    # apparently it is faster to go through numpy
    array = np.array(encoded_list, dtype=np.float32)
    array = torch.tensor(array)

    # edges are bidirectional
    # so need to concat the edge array with itself
    if arrayType == EncoderType.EDGE:
        num_edges, _ = array.shape
        array = torch.concat([array, array])

        # add a one hot encoding of edge direction
        forward_edge_enc = torch.cat((torch.ones(num_edges, 1), torch.zeros(num_edges, 1)), dim=0)
        backward_edge_enc = torch.cat((torch.zeros(num_edges, 1), torch.ones(num_edges, 1)), dim=0)

        array = torch.cat((array, forward_edge_enc, backward_edge_enc), dim=1)

    return array


def make_graph_arrays(encoders, graph):
    node_array = get_attr_array(encoders, graph.nodes(), EncoderType.NODE)
    edge_array = edges_list_to_edge_array(graph.edges(), graph.nodes())
    edge_attr_array = get_attr_array(encoders, graph.edges(), EncoderType.EDGE)

    return node_array, edge_array, edge_attr_array

def make_bb_id_list(graph):
    bb_list = []
    for node in graph.nodes():
        bbID = int(node.attr["bbID"]) - 1
        bb_list.append(bbID)

    bb_list = torch.tensor(bb_list, dtype=torch.int64)
    return bb_list

class CFG():
    def __init__(self, cfg_edge_index, num_bbs, bb_batch):
        self.cfg_edge_index = cfg_edge_index
        self.num_bbs = num_bbs
        self.bb_batch = bb_batch

def make_cfg(kernel_data, invocation, apply_directives, i=0):
    # the kernels are stored with pragma location labels
    # so apply directives is used to get a cpp file with no labels
    out = f"temp{i}.cpp"

    # apply_directives(kernel_data, -1, out)
    apply_directives(kernel_data, i, out)


    # get a normal graph of the kernelp
    full_invocation = invocation + f" --top {kernel_data.kernel_name} --src {out} --datasetIndex 0"
    graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)

    # print(graphOutput)

    graph = pgv.AGraph(string=graphOutput.stdout)

    os.remove(out)

    # and then build CFG from it

    return make_cfg_from_graph(graph)

def make_cfg_from_graph(graph):
    # this is not the most elegant solution, 
    # possibly there should be some kind of consolidated meta-data from the c++ graph compiler
    # but its not much effort to reverse engineer
    
    num_bbs = 0
    node_bb_dict = {}

    for node in graph.nodes():
        # get ID
        bbID = int(node.attr["bbID"])

        # store mapping from node name to bbID
        # to use with edges
        node_bb_dict[node] = bbID -1
        
        num_bbs = max(num_bbs, bbID)



    # get edges between basic blocks
    bb_edge_list = defaultdict(set)

    for edge in graph.edges():
        # we don't want BBs connected by dataflow or memory element use
        if not (edge.attr["flowType"] == "control" or edge.attr["flowType"] == "call"):
            continue
            
        # get the nodes that the edge connects to
        node_a = edge[0]
        node_b = edge[1]

        # get the BB ID 
        source_bb = node_bb_dict[node_a]
        target_bb = node_bb_dict[node_b]

        # if edge.attr["dir"] == "back":
        #     temp = target_bb
        #     target_bb = source_bb
        #     source_bb = temp

        # if there is control flow between 2 different BBs
        if source_bb != target_bb:
            # if there is not an edge already registered from target to source
            if source_bb not in bb_edge_list[target_bb]:
                # register an edge from source to target
                bb_edge_list[source_bb].add(target_bb)


    # make two lists, one of edge sources, one of edge targets
    source_list = []
    target_list = []

    G = pgv.AGraph(directed=True)
    for source in bb_edge_list:
        for target in bb_edge_list[source]:
            # forward edge
            source_list.append(source)
            target_list.append(target)

            # backward edge
            source_list.append(target)
            target_list.append(source)

            G.add_edge(source, target)

    # G.layout(prog='dot')  # Use the DOT layout engine
    # G.draw("cfg.png", format='png')

    # make a (2,N) array
    cfg_edge_index = np.array([source_list, target_list], dtype=np.int64)
    cfg_edge_index = torch.from_numpy(cfg_edge_index)          

    # bb_batch is used to aggregate bb embeddings when using batching
    # its incremented by 1 each time the batcher pulls a new kernel
    # so should start at 0
    bb_batch = torch.zeros(num_bbs, dtype=torch.int64)

    cfg = CFG(cfg_edge_index, num_bbs, bb_batch)

    return cfg