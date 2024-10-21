from torch_geometric.loader import DataLoader

import torch
from tqdm import tqdm

import pandas as pd

import balorgnn.generate.output_config as outConf

import re

def process_metrics(output_config, shift, out, batch, batch_index, row):
    for i, metric in enumerate(output_config.metrics):    
    # for i, metric in enumerate(['LUTs join all','FFs join all','DSPs join all','BRAMs join all','Latency join all','Valid join']):
        metric_index = i + shift

        if (metric, metric_index) not in out:  
            print(f"Issue with metric indices {(metric, metric_index)} not in output") 
            break

        ground_truth = batch.y[batch_index, metric_index]
        inferred = out[(metric, metric_index)][batch_index].squeeze()

        # metric = re.sub(" join", "", metric)

        real_ground_truth = output_config.unnormalize(metric, ground_truth).item()
        real_inferred = output_config.unnormalize(metric, inferred).item()
        
        row[f"real {metric}"] = real_ground_truth
        row[f"inferred {metric}"] = real_inferred

        row[f"real {metric} normal"] = ground_truth.item()
        row[f"inferred {metric} normal"] = inferred.item()

def inference(trainer, dataset, out_name, epoch, gpu_id, combine_vast):
    # print("hardcoded use join")
    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = trainer.model

    model.eval()
    output_data = []
    with torch.no_grad():
        for batch in tqdm(loader):  
            batch = batch.to(device)


            out, _, _ = model.to(device)(batch)

            for batch_index in range(len(batch.kernel)):
                row = {}
                row["kernel"] = batch.kernel[batch_index]
                row["pragmas"] = batch.pragmas[batch_index]
                output_config_name = batch.output_config_name[batch_index].item()
                row["output_config"] = output_config_name

                output_config = batch.output_config[batch_index]

                # initialize
                output_config.set_functions()

                shift = batch.shift[batch_index].item()
                process_metrics(output_config, shift, out, batch, batch_index, row)

                # if combine_vast:
                #     shift = batch.join_shift[batch_index].item()
                #     shift = 39
                #     # print(shift)

                #     output_config = outConf.OutputConfigVast(join=True)
                #     process_metrics(output_config, shift, out, batch, batch_index, row)
                output_data.append(row)
    model.train()
    df = pd.DataFrame(output_data).fillna(0)

    df.to_csv(f"{trainer.results_dir}/{out_name}_{epoch}.csv")

