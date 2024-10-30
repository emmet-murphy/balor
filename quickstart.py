import argparse
import subprocess
import shutil
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_dataset', action='store_true', help='Download more than 37,000 DB4HLS files already encoded and ready for use.')

    parser.add_argument('--download_pretrained', action='store_true', help='Download our best model to use for evaluation.')
    
    args = parser.parse_args()

    if args.download_dataset:
        url = "https://polybox.ethz.ch/index.php/s/FIi8xmd5jFr8XZA/download"
        download_command = ["wget", "--content-disposition", url]
        subprocess.run(download_command)

        top_folder = "balorgnn/datasets/" 
        zip_file = "db4hls_download.zip"

        os.makedirs(top_folder, exist_ok=True)

        unzip_command = ["unzip", zip_file, "-d", top_folder]
        subprocess.run(unzip_command)

        os.remove(zip_file)

    if args.download_pretrained:
        model_weights_url = "https://polybox.ethz.ch/index.php/s/qm0E347v3pKpJzL/download"
        download_command = ["wget", "--content-disposition", model_weights_url]
        subprocess.run(download_command)

        top_folder = "balorgnn/outputs/model_weights/db4hls_pretrained/limerick/snake/0" 

        os.makedirs(top_folder, exist_ok=True)

        source_file = "580.pth" 
        destination_file = os.path.join(top_folder, source_file)

        shutil.copy(source_file, destination_file)

        os.remove(source_file)

        val_url = "https://polybox.ethz.ch/index.php/s/I6KpCnYXwrbQSMU/download"

        download_command = ["wget", "--content-disposition", val_url]
        subprocess.run(download_command)

        top_folder = "balorgnn/outputs/results/db4hls_pretrained/limerick/snake/0" 

        os.makedirs(top_folder, exist_ok=True)

        source_file = "val_580.csv" 
        destination_file = os.path.join(top_folder, source_file)

        shutil.copy(source_file, destination_file)

        os.remove(source_file)

        test_url = "https://polybox.ethz.ch/index.php/s/yHGF25PfS7tbPHD/download"

        download_command = ["wget", "--content-disposition", test_url]
        subprocess.run(download_command)

        top_folder = "balorgnn/outputs/results/db4hls_pretrained/limerick/snake/0" 

        os.makedirs(top_folder, exist_ok=True)

        source_file = "test_580.csv" 
        destination_file = os.path.join(top_folder, source_file)

        print(source_file, destination_file)
        shutil.copy(source_file, destination_file)

        os.remove(source_file)