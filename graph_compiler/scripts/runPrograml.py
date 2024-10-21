import programl
import argparse
import subprocess

def run_programl(relative_path, input_file):
    graph = programl.from_clang([relative_path + input_file, "-DMY_MACRO=3"])
    output = 'outputs/' + input_file + '_realPrograml'

    with open(output + '.dot', 'w') as f:
        f.write(programl.to_dot(graph))

def main():
    parser = argparse.ArgumentParser(description="Run programl on .cpp file")
    parser.add_argument("input_file", help="input file name")
    parser.add_argument("relative_path", help="Path to the input file folder")

    args = parser.parse_args()
    run_programl(args.relative_path, args.input_file)

if __name__ == "__main__":
    main()

