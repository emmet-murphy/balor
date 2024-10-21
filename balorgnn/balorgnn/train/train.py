
import argparse
import os
import torch
from torch_geometric.loader import DataLoader

import time

from balorgnn.data.dataset import CustomDataset
from balorgnn.train.models import CatArch, BullArch, RhinoArch, MouseArch, SnakeArch, CamelArch, DogArch

from tqdm import tqdm

from balorgnn.train.inference import inference

from collections import defaultdict

from enum import Enum

import random

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

import json

class Architectures(Enum):
    CAT = 1
    BULL = 2
    SNAKE = 3
    RHINO = 4
    MOUSE = 5
    CAMEL = 6
    DOG = 7

def process_data(i, dataset):
    data = dataset[i]
    if data.output_config_name == 5:
        if random.random() < 0.8:
            return 'train', i
        else:
            return 'valid', i
    else:
        return 'train', i
    
class Trainer:
    def __init__(self, dataset, architecture, output_base_path, output_tail_path, gpu_id, use_cpu, use_test_set):
        self.gpu_id = gpu_id
        self.dataset = dataset

        self.initialize_folders(output_base_path, output_tail_path)

        # allow manually specifying not to use the GPU
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        # split the data the same every time
        generator1 = torch.Generator().manual_seed(42)


        random.seed(42)
        if use_test_set:
            self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15],generator=generator1)
            self.test_data = self.test_data[:]

            # random split returns "subset" objects, but I have a custom dataset type I need to get back to
            # there's other people complaining about this conversion being awkward online
            # but I can't see any other way to do it either
            self.train_data = self.train_data[:]
            self.val_data = self.val_data[:]
        else:
            train_indices = []
            valid_indices = []


            with ProcessPoolExecutor(max_workers=12) as executor:
                futures = []
                for i in range(len(self.dataset)):
                    futures.append(executor.submit(process_data, i, self.dataset))
                    
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating data split"):
                    result_type, i = future.result()
                    if result_type == 'train':
                        train_indices.append(i)
                    else:
                        valid_indices.append(i)

                write_indices = {"train" : train_indices, 'valid' : valid_indices}

                with open("indices.json", 'w') as f:
                    json.dump(write_indices, f)

            # with open("indices.json", 'r') as f:
            #     data = json.load(f)

            # Accessing the lists
            # train_indices = data['train']
            # valid_indices = data['valid']

            self.train_data = torch.utils.data.Subset(self.dataset, train_indices)
            self.val_data = torch.utils.data.Subset(self.dataset, valid_indices)

            self.train_data = self.train_data[:]
            self.val_data = self.val_data[:]
    

        print(f'{len(dataset)} graphs in total:')
        if use_test_set:
            print(f'train: {len(self.train_data)}, val:  {len(self.val_data)}, test: {len(self.test_data)}')
        else: 
            print(f'train: {len(self.train_data)}, val:  {len(self.val_data)}')
        batch_size = 32
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, drop_last=True)


        self.outputs = list(self.train_data[0].all_outputs)
        output_config_name = self.train_data[0].output_config_name.item()


        self.num_features = self.train_loader.dataset[0].num_features
        self.edge_dim = self.train_loader.dataset[0].edge_attr.shape[1]
        if architecture == Architectures.CAT:
            self.model = CatArch(self.num_features, self.edge_dim, self.outputs, output_config_name).to(self.device)
        if architecture == Architectures.BULL:
            self.model = BullArch(self.num_features, self.edge_dim, self.outputs).to(self.device)
        if  architecture == Architectures.RHINO:
            self.model = RhinoArch(self.num_features, self.edge_dim, self.outputs).to(self.device)
        if architecture == Architectures.SNAKE:
            self.model = SnakeArch(self.num_features, self.edge_dim, self.outputs).to(self.device)
        if architecture == Architectures.MOUSE:
            self.model = MouseArch(self.num_features, self.edge_dim, self.outputs).to(self.device)
        if architecture == Architectures.CAMEL:
            self.model = CamelArch(self.num_features, self.edge_dim, self.outputs, output_config_name).to(self.device)
        if architecture == Architectures.DOG:
            self.model = DogArch(self.num_features, self.edge_dim, self.outputs).to(self.device)

    def initialize_folders(self, output_base_path, output_tail_path):
        if output_base_path.endswith("/"):
            output_base_path = output_base_path[:-1]

        if output_tail_path.startswith("/"):
            output_tail_path = output_tail_path[1:]
        if output_tail_path.endswith("/"):
            output_tail_path = output_tail_path[:-1]

        self.model_dir = f"{output_base_path}/model_weights/{output_tail_path}/"
        self.results_dir = f"{output_base_path}/results/{output_tail_path}/"

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def train_main(self, num_epochs, lr, save_train, use_test_set, combine_vast):
        epochs = range(num_epochs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        for epoch in epochs:
            print(f'Epoch {epoch} train')
            _, loss_dict_train = self.run_epoch()

            print((f'Train loss breakdown {loss_dict_train}'))

            if (epoch + 1) % 10 == 0:
                self.save(epoch + 1, save_train, use_test_set, combine_vast)


    def run_epoch(self):
        self.model.train()
        total_loss = 0



        loss_dict = defaultdict(float)
        # for t in self.outputs:
        #     loss_dict[t] = 0.0
        
        startTime = time.time()
        for data in tqdm(self.train_loader):
            
            data = data.to(self.device)
            self.optimizer.zero_grad()
            _, loss, loss_dict_ = self.model.to(self.device)(data)
            loss.backward()

            total_loss += loss.item() * data.num_graphs


            for t in loss_dict_:
                loss_dict[t] += loss_dict_[t].item()
            
            self.optimizer.step()
        endTime = time.time()
        print(f"Epoch Training Time: {endTime-startTime}")
        return total_loss / len(self.train_loader.dataset), {key: v / len(self.train_loader) for key, v in loss_dict.items()}
   
    def save(self, epoch, save_train, use_test_set, combine_vast):
        torch.save(self.model.state_dict(),f"{self.model_dir}/{epoch}.pth")

        inference(self, self.val_data, "val",  epoch, self.gpu_id, combine_vast)

        if use_test_set:
            inference(self, self.test_data, "test", epoch, self.gpu_id, combine_vast)
        if save_train:
            inference(self, self.train_data, "train", epoch, self.gpu_id, combine_vast)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN-based QoR estimation")
    parser.add_argument("--data_dir", required=True, help="Folder of .pth files, 1 per graph")

    parser.add_argument("--gpu_id", default=0, help="Which GPU to run on. Ignored if cuda is not present")
    parser.add_argument("--use_cpu", default=False, help="Run on CPU even on a cuda-capable system")

    parser.add_argument("--output_base_path", required=True, help="Where to place model_weight, result, and training error dirs")
    parser.add_argument("--output_tail_path", required=True, help="Folder tree location inside of model_weight, result and training dirs")

    parser.add_argument("--num_epochs", default=1000)
    parser.add_argument("--lr", default=0.0005)

    parser.add_argument('--save_train', action='store_true')
    parser.add_argument('--skip_test_set', action='store_false')

    parser.add_argument('--combine_vast', action='store_true')



    archGroup = parser.add_mutually_exclusive_group(required=True)
    archGroup.add_argument('--arch_cat', action='store_true', help='Run with Arch. Cat, the baseline architecture with 6 message-passing layers')
    archGroup.add_argument('--arch_bull', action='store_true', help='Run with Arch. Bull, the reduced architecture with only 2 message-passing layers')
    archGroup.add_argument("--arch_rhino", action='store_true', help='Run with Arch. Rhino, the flat enhanced architecture with 2 message-passing layers and 2 residual blocks')
    archGroup.add_argument("--arch_snake", action='store_true', help='Run with Arch. Snake, the basic hierarchical architecture')
    archGroup.add_argument("--arch_mouse", action='store_true', help='Run with Arch. Mouse, the hierarchical architecture with more res blocks')
    archGroup.add_argument("--arch_camel", action='store_true', help='Run with Arch. Camel, the hierarchical architecture with 3 action level mp layers')
    archGroup.add_argument("--arch_dog", action='store_true', help='Run with Arch. DOG, the hierarchical architecture with 2 action level mp layers, 128 feature size')



    args = parser.parse_args()

    dataset = CustomDataset(args.data_dir)

    architecture = None
    if args.arch_cat:
        architecture = Architectures.CAT
    elif args.arch_bull:
        architecture = Architectures.BULL
    elif args.arch_rhino:
        architecture = Architectures.RHINO
    elif args.arch_snake:
        architecture = Architectures.SNAKE
    elif args.arch_mouse:
        architecture = Architectures.MOUSE
    elif args.arch_camel:
        architecture = Architectures.CAMEL
    elif args.arch_dog:
        architecture = Architectures.DOG

    assert(architecture is not None)

    trainer = Trainer(dataset, architecture, args.output_base_path, args.output_tail_path, args.gpu_id, args.use_cpu, args.skip_test_set)
    trainer.train_main(args.num_epochs, args.lr, args.save_train, args.skip_test_set, args.combine_vast)



