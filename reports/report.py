

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
import os

def process_df(path, metrics):
    df = pd.read_csv(path)

    # convert weird metric names
    # TODO: rewrite code to use proper metric names to begin with
    for metric in metrics:
        df[f"real {metric}"] = df[f"real {metric} join all"]
        df[f"inferred {metric}"] = df[f"inferred {metric} join all"]

    return df

def calculate_metrics(output_folder, epoch):
    csv_path = f"{output_folder}/val_{epoch}.csv"   

    df = pd.read_csv(csv_path)

    # inspired by the UCLA Vast papers (GNN-DSE, HARP, PROG SG, HLSYN, etc),
    # we choose the best model by RMSE on the normalized output of the model
    metrics = ["LUTs", "FFs", "DSPs", "BRAMs", "Latency", "Clock"]
    
    rmse = {}
    for metric in metrics:
        rmse[metric] = np.sqrt(np.mean((df[f"real {metric} normal"] - df[f"inferred {metric} normal"])**2))

    return rmse


def find_best_model(output_folder):
    min_sum = None
    for epoch in range(10, 1000 + 10, 10):
        if not os.path.isfile(f"{output_folder}/val_{epoch}.csv"):
            continue

        sum = 0
        rmse = calculate_metrics(output_folder, epoch)
        for metric in rmse:
            sum = sum + rmse[metric]

        if min_sum is None or sum < min_sum:
            min_epoch = epoch
            min_sum = sum

    selection_metric = min_sum / len(rmse)

    print(f"Best model on validation set: Epoch {min_epoch} with a selection metric of {selection_metric}")
    print(f"Selection metric is average RMSE when inputs are scaled from -1 to 1")
    return min_epoch

def report_single(dataset, id=0):
    epoch = find_best_model(f"balorgnn/outputs/results/{dataset}/limerick/snake/{id}")



