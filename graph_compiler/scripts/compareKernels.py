import argparse
import pygraphviz as pgv
import os
import subprocess

def count_nodes(graph, label_attribute, multByEdges):
    node_count = {}
    total = 0
    for node in graph.nodes():
        label = node.attr[label_attribute]

        if label.endswith('*'):
            label = 'i64'
        if label.endswith(']'):
            label = 'i64'

        # Check if the node has a diamond shape
        is_diamond = 'shape' in node.attr and node.attr['shape'] == 'diamond'

        updateVal = 1
        # Update node count based on the shape
        if is_diamond and multByEdges:
            updateVal = len(list(graph.out_edges(node)))


        node_count[label] = node_count.get(label, 0) + updateVal
        total = total + updateVal
    node_count["total"] = total
    return node_count

def compare_kernel_graphs(dot_file1, dot_file2, doPrint):
    graph1 = pgv.AGraph(dot_file1)
    graph2 = pgv.AGraph(dot_file2)

    node_count1 = count_nodes(graph1, "label", True)
    node_count2 = count_nodes(graph2, "keyText", False)

    all_keys = set(node_count1.keys()).union(set(node_count2.keys()))

    if(doPrint):
        print("Node Counts Comparison:")
        print("{:<15} {:<15} {:<15} {:<15}".format("Node Type", "Graph 1 Count", "Graph 2 Count", "Difference"))
        print("-" * 60)

    all_differences_zero = True

    for key in sorted(all_keys):
        count1 = node_count1.get(key, 0)
        count2 = node_count2.get(key, 0)
        difference = count2 - count1
        if difference != 0:
            all_differences_zero = False
        if(doPrint):
            print("{:<15} {:<15} {:<15} {:<15}".format(key, count1, count2, difference))
    
    total1 = node_count1.get("total", 0)
    total2 = node_count2.get("total", 0)

    if(doPrint):
        print("{:<15} {:<15} {:<15} {:<15}".format("Total", total1, total2, total2-total1))
    
    return all_differences_zero


from runPrograml import run_programl

def get_subfolder_names(folder_path):
    # Get a list of all items (files and subfolders) in the given folder
    all_items = os.listdir(folder_path)

    # Filter out only the subfolders
    subfolders = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]

    return subfolders

def generate_kernal_graphs(kernel, kernel_file):
    folder = "machsuite/" + kernel + "/"

    assert os.path.exists(folder + kernel_file), folder + kernel_file

    # Example 1: Run the script without capturing output
    command = ["./bin/AIR" , "--top", kernel, "--src", folder + kernel_file, "--hide_values", "--proxy_programl", "--make_dot", "--one_hot_types", "--absorb_types"]

    subprocess.run(command, capture_output=True)

    run_programl(folder, kernel_file)

def main(kernel):
    subfolders = get_subfolder_names("machsuite")

    if kernel is None:
        kernels = subfolders
    else:
        kernels = [kernel]

    for kernel in kernels:
        kernel_file = kernel + '.cpp'
        
        generate_kernal_graphs(kernel, kernel_file)

        dotfile1 = "outputs/" + kernel_file + "_realPrograml.dot"
        dotfile2 = "outputs/" + kernel + ".dot"
        print("Kernel: " + kernel)
        no_difference = compare_kernel_graphs(dotfile1, dotfile2, True)
        if no_difference:
            print("0 discrepencies")

        else:
            kernel_file = kernel + "_no_reuse.cpp"
            generate_kernal_graphs(kernel, kernel_file)
            dotfile1 = "outputs/" + kernel_file + "_realPrograml.dot"
            dotfile2 = "outputs/" + kernel + ".dot"
            no_difference = compare_kernel_graphs(dotfile1, dotfile2, False)

            if no_difference:
                print("0 discrepencies once reuse removed")
            else:
                compare_kernel_graphs(dotfile1, dotfile2, True)
        print("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare kernel graphs')
    parser.add_argument('--kernel', default=None)

    args = parser.parse_args()

    main(args.kernel)
