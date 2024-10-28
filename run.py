import argparse
import subprocess

def get_base_generate_command():

    generate_command = ["python", "balorgnn/balorgnn/generate/generate_dataset.py"]
    # specify datasets folder
    generate_command.append("--dataset_folder")
    generate_command.append("balorgnn/datasets")

    # specify graph_compiler
    generate_command.append("--graph_compiler")
    generate_command.append("graph_compiler/bin/graph_compiler")

    generate_command.append("--inputs_folder")
    generate_command.append("balorgnn/inputs")

    # specify graph config
    generate_command.append("--graph_config_limerick")

    # add the combine vast flag (this flag needs to be rewritten to make more sense)
    generate_command.append("--combine_vast")

    return generate_command


def train(dataset):
    train_command = ["python", "balorgnn/balorgnn/train/train.py"]

    # specify dataset directory
    train_command.append("--data_dir")
    train_command.append(f"balorgnn/datasets/{dataset}/limerick/")

    # specify output directory
    train_command.append("--output_base_path")
    train_command.append("balorgnn/outputs")

    # specify output subfolders
    train_command.append("--output_tail_path")
    train_command.append(f"{dataset}/limerick/snake/0")

    # perform inference on the train set and save results every 10 epochs
    train_command.append("--save_train")

    # use the snake architecture
    train_command.append("--arch_snake")

    subprocess.run(train_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_db4hls', action='store_true', help='Convert 36,000 DB4HLS files to pytorch geometric files. Requires the graph compiler to be built, and the DB4HLS mysql server to be present.')

    parser.add_argument('--train_db4hls', action='store_true', help='Train the mode. Generate must be ran first.')

    args = parser.parse_args()

    if args.generate_db4hls:
        print("Generating DB4HLS dataset")

        generate_command = get_base_generate_command()

        # add DB4HLS kernels
        generate_command.append("--kernels_red")

        # specify folder name in datasets
        generate_command.append("--output_folder")
        generate_command.append("db4hls")
       
        subprocess.run(generate_command)


    if args.train_db4hls:
        print("Training on DB4HLS dataset")
        train("db4hls")

