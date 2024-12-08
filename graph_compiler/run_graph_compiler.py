import argparse
import subprocess

def compile_graph(mode, src, top, make_pdf, generalize_types):
    # Base command
    command = ["./bin/graph_compiler", "--src", src, "--top", top, "--datasetIndex", "NA", "--graphType", "NA"]

    if make_pdf:
        command += ["--make_pdf"]

    if not generalize_types:
        command += ["--one_hot_types"]

    if mode == "base":
        command += [
                    "--proxy_programl",
                    ]
    elif mode == "opt":
        command += [
                    "--allocas_to_mem_elems",
                    "--remove_sexts",
                    "--remove_single_target_branches",
                    "--drop_func_call_proc",
                    "--absorb_types",
                    "--absorb_pragmas",
                    ]
    else:
        raise ValueError(f"Unknown mode: {mode}")


    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    print(" ".join(command))

    if result.stderr:
       print(result.stderr)

    # Print the output
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Graph Compiler Script")
    
    # Two-way choice between baseline or optimized
    parser.add_argument("--mode", choices=["base", "opt"], 
                        help="Choose compilation mode: baseline or optimized",
                        required=True)
    
    # Source file argument
    parser.add_argument("--src", help="Path to the source file", required=True)
    parser.add_argument("--top", help="Top level function", required=True)
    parser.add_argument("--make_pdf", action="store_true", help="Print pdf in output folder")
    parser.add_argument("--generalize_types", action="store_true", help="Output characteristics of types rather than one-hot-encoding them")
    
    args = parser.parse_args()

    # Call the compile function with the parsed arguments
    compile_graph(args.mode, args.src, args.top, args.make_pdf, args.generalize_types)

if __name__ == "__main__":
    main()
